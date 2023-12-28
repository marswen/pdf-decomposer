import random
import numpy as np
import pandas as pd

POPULATION_SIZE = 100
GENERATIONS = 100

ELITE_SIZE = 20
CROSSOVER_SIZE = 40
MUTATION_SIZE = 80

CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3

EARLY_STOP_TURNS = 10
MIN_OVERLAP = 0.3


def calculate_box_containment(box1, box2, axis=None):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    intersection_x1 = max(x11, x21)
    intersection_y1 = max(y11, y21)
    intersection_x2 = min(x12, x22)
    intersection_y2 = min(y12, y22)
    if axis == 0:
        intersection_height = max(0, intersection_y2 - intersection_y1 + 1)
        box1_height = (y12 - y11 + 1)
        box2_height = (y22 - y21 + 1)
        containment = intersection_height / min(box1_height, box2_height)
    elif axis == 1:
        intersection_width = max(0, intersection_x2 - intersection_x1 + 1)
        box1_width = (x12 - x11 + 1)
        box2_width = (x22 - x21 + 1)
        containment = intersection_width / min(box1_width, box2_width)
    else:
        intersection_area = (max(0, intersection_x2 - intersection_x1 + 1) *
                             max(0, intersection_y2 - intersection_y1 + 1))
        box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
        containment = intersection_area / min(box1_area, box2_area)
    return containment


def calculate_line_containment(text_box_lines, axis):
    if axis == 0:
        lines_x1 = [min([x[1] for x in line]) for line in text_box_lines]
        lines_x2 = [max([x[3] for x in line]) for line in text_box_lines]
        lines = [pair for pair in zip(lines_x1, lines_x2)]
    else:
        lines_y1 = [min([x[0] for x in line]) for line in text_box_lines]
        lines_y2 = [max([x[2] for x in line]) for line in text_box_lines]
        lines = [pair for pair in zip(lines_y1, lines_y2)]
    containment = list()
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection1 = max(lines[i][0], lines[j][0])
            intersection2 = min(lines[i][1], lines[j][1])
            intersection = max(0, intersection2 - intersection1 + 1)
            box1_height = (lines[i][1] - lines[i][0] + 1)
            box2_height = (lines[j][1] - lines[j][0] + 1)
            containment.append(intersection / min(box1_height, box2_height))
    return np.mean(containment) + 1e-10


def calculate_table_box_containment(text_boxes):
    row_box_containment = np.zeros((len(text_boxes), len(text_boxes)))
    col_box_containment = np.zeros((len(text_boxes), len(text_boxes)))
    for i in range(len(text_boxes)):
        for j in range(i + 1, len(text_boxes)):
            row_box_containment[i, j] = row_box_containment[j, i] = calculate_box_containment(
                text_boxes[i], text_boxes[j], axis=0)
            col_box_containment[i, j] = col_box_containment[j, i] = calculate_box_containment(
                text_boxes[i], text_boxes[j], axis=1)
    return row_box_containment, col_box_containment


def dfs(node, visited, cluster, adj_matrix, threshold):
    visited.append(node)
    cluster.append(node)
    for neighbor in np.nonzero(adj_matrix[node] >= threshold)[0].tolist():
        if neighbor not in visited:
            dfs(neighbor, visited, cluster, adj_matrix, threshold)


def cluster_boxes(adj_matrix, threshold):
    visited = list()
    clusters = list()
    for i in range(len(adj_matrix)):
        if i not in visited:
            cluster = []
            dfs(i, visited, cluster, adj_matrix, threshold)
            clusters.append(cluster)
    return clusters


def merge_close_column(table, text_boxes):
    table_box = table.applymap(
        lambda cell_ids: sorted([text_boxes[idx] for idx in cell_ids], key=lambda x: (x[1], x[0]))
        if isinstance(cell_ids, list) else cell_ids)
    inner_gaps = list()
    for row in table_box.values.tolist():
        for cell in row:
            if isinstance(cell, list) and len(cell) > 1:
                for i in range(len(cell) - 1):
                    if cell[i][2] < cell[i + 1][0]:
                        inner_gaps.append(cell[i + 1][0] - cell[i][2])
    if len(inner_gaps) > 0:
        mean_inner_gap = np.mean(inner_gaps)
        table_box = table_box.applymap(lambda x: np.array(x) if isinstance(x, list) else x)
        merge_table_box = table_box.applymap(
            lambda x: np.min(x[:, :2], axis=0).tolist() + np.max(x[:, 2:], axis=0).tolist()
            if isinstance(x, np.ndarray) else x)
        close_cells = list()
        for i_row, row in merge_table_box.iterrows():
            last_box = None
            for i_cell, cell in row.items():
                if isinstance(cell, list):
                    if last_box is None:
                        last_box = cell
                    else:
                        gap = cell[0] - last_box[2]
                        if gap < mean_inner_gap:
                            close_cells.append([i_row, i_cell])
                        last_box = cell
        for cell in close_cells:
            for i in range(cell[1] - 1, -1, -1):
                if isinstance(table.iloc[cell[0], i], list):
                    table.iloc[cell[0], i] = table.iloc[cell[0], i] + table.iloc[cell[0], cell[1]]
                    table.iloc[cell[0], cell[1]] = np.nan
                    break
    return table


def create_table(text_boxes, text_recs, row_clusters, col_clusters):
    row_clusters = sorted(row_clusters, key=lambda row_ids: min([text_boxes[x][1] for x in row_ids]))
    col_clusters = sorted(col_clusters, key=lambda col_ids: min([text_boxes[x][0] for x in col_ids]))
    cell_dict = {}
    for i, row in enumerate(row_clusters):
        for value in row:
            cell_dict[value] = [i]
    for i, col in enumerate(col_clusters):
        for value in col:
            cell_dict[value].append(i)
    table = pd.DataFrame(columns=range(len(col_clusters)), index=range(len(row_clusters)))
    for value, (r, c) in cell_dict.items():
        if isinstance(table.iloc[r, c], list):
            table.iloc[r, c].append(value)
        else:
            table.iloc[r, c] = [value]
    table = merge_close_column(table, text_boxes)
    table = table.applymap(lambda x: ' '.join([text_recs[idx] for idx in x]).strip() if isinstance(x, list) else x)
    table.replace('', np.nan, inplace=True)
    table.dropna(how='all', axis=1, inplace=True)
    return table


class Individual:
    def __init__(self, text_boxes, genes, row_box_containment, col_box_containment):
        self.text_boxes = text_boxes
        self.genes = genes
        self.row_box_containment = row_box_containment
        self.col_box_containment = col_box_containment
        self.unique_row_containment = np.unique(row_box_containment.flatten())
        self.unique_col_containment = np.unique(col_box_containment.flatten())
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        row_clusters = cluster_boxes(self.row_box_containment, self.genes[0])
        col_clusters = cluster_boxes(self.col_box_containment, self.genes[1])
        row_line_containment = calculate_line_containment(
            [[self.text_boxes[x] for x in cluster] for cluster in row_clusters], axis=0)
        col_line_containment = calculate_line_containment(
            [[self.text_boxes[x] for x in cluster] for cluster in col_clusters], axis=1)
        return 1 / (row_line_containment + col_line_containment) / (len(row_clusters) * len(col_clusters))

    def crossover(self, partner):
        child_genes = []
        for i in range(len(self.genes)):
            if random.random() < CROSSOVER_RATE:
                child_genes.append(partner.genes[i])
            else:
                child_genes.append(self.genes[i])
        return Individual(self.text_boxes, child_genes, self.row_box_containment, self.col_box_containment)

    def mutate(self):
        child_genes = []
        if random.random() < MUTATION_RATE:
            child_genes.append(random.choice(self.unique_row_containment[self.unique_row_containment > MIN_OVERLAP]))
        else:
            child_genes.append(self.genes[0])
        if random.random() < MUTATION_RATE:
            child_genes.append(random.choice(self.unique_col_containment[self.unique_col_containment > MIN_OVERLAP]))
        else:
            child_genes.append(self.genes[1])
        return Individual(self.text_boxes, child_genes, self.row_box_containment, self.col_box_containment)


def initialize_population(text_boxes, row_box_containment, col_box_containment):
    unique_row_containment = np.unique(row_box_containment.flatten())
    unique_col_containment = np.unique(col_box_containment.flatten())
    unique_row_containment = unique_row_containment[unique_row_containment > MIN_OVERLAP]
    unique_col_containment = unique_col_containment[unique_col_containment > MIN_OVERLAP]
    row_col_containment_compos = [(r, c) for r in unique_row_containment for c in unique_col_containment]
    candidate_genes = random.sample(row_col_containment_compos, k=min(len(row_col_containment_compos), POPULATION_SIZE))
    population = [Individual(text_boxes, genes, row_box_containment, col_box_containment) for genes in candidate_genes]
    return population


def elitism_selection(population):
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    elite = sorted_population[:ELITE_SIZE]
    return elite


def crossover_population(mating_pool):
    offspring = []
    for i in range(CROSSOVER_SIZE):
        parents = random.sample(mating_pool, 2)
        child = parents[0].crossover(parents[1])
        offspring.append(child)
    return offspring


def mutate_population(mating_pool):
    mutations = list()
    for i in range(MUTATION_SIZE):
        parent = random.choice(mating_pool)
        child = parent.mutate()
        mutations.append(child)
    return mutations


def run_genetic_algorithm(text_boxes, row_box_containment, col_box_containment):
    population = initialize_population(text_boxes, row_box_containment, col_box_containment)
    turns_no_change = 0
    last_fitness = 0
    best_individual = max(population, key=lambda x: x.fitness)
    for generation in range(GENERATIONS):
        mating_pool = elitism_selection(population)
        offspring = crossover_population(mating_pool)
        mutation = mutate_population(mating_pool)
        population = mating_pool + offspring + mutation
        best_individual = max(population, key=lambda x: x.fitness)
        if best_individual.fitness == last_fitness:
            turns_no_change += 1
        last_fitness = best_individual.fitness
        if turns_no_change == EARLY_STOP_TURNS:
            break
    return best_individual


def strict_clarify(text_boxes, row_box_containment, col_box_containment):
    col_clusters = cluster_boxes(col_box_containment, 1)
    while True:
        col_spans = [[text_boxes[x] for x in cluster] for cluster in col_clusters]
        lines_y1 = [min([x[0] for x in line]) for line in col_spans]
        lines_y2 = [max([x[2] for x in line]) for line in col_spans]
        lines = sorted([pair for pair in zip(lines_y1, lines_y2)], key=lambda x: x[0])
        intersections = list()
        for i in range(len(lines) - 1):
            intersection1 = max(lines[i][0], lines[i + 1][0])
            intersection2 = min(lines[i][1], lines[i + 1][1])
            intersections.append(max(0, intersection2 - intersection1 + 1))
        bad_intersections = [i for i, x in enumerate(intersections) if x > 0]
        if len(bad_intersections) > 0:
            bad_cols = set([x + 1 for x in bad_intersections] + bad_intersections)
            bad_col = sorted(bad_cols, key=lambda x: len(col_spans[x]))[0]
            col_clusters = [x for i, x in enumerate(col_clusters) if i != bad_col]
        else:
            break
    remain_boxes = [i for x in col_clusters for i in x]
    for i in range(len(row_box_containment)):
        if i not in remain_boxes:
            row_box_containment[i, :] = row_box_containment[:, i] = 0
    best_fitness = 0
    best_row_clusters = [remain_boxes]
    unique_row_containment = np.unique(row_box_containment.flatten())
    unique_row_containment = unique_row_containment[unique_row_containment > MIN_OVERLAP]
    for threshold in unique_row_containment:
        row_clusters = cluster_boxes(row_box_containment, threshold)
        row_clusters = [[i for i in row if i in remain_boxes] for row in row_clusters]
        row_clusters = [row for row in row_clusters if len(row) > 0]
        row_line_containment = calculate_line_containment(
            [[text_boxes[x] for x in cluster] for cluster in row_clusters], axis=0)
        fitness = 1 / row_line_containment / (len(row_clusters))
        if fitness > best_fitness:
            best_fitness = fitness
            best_row_clusters = row_clusters
    return best_row_clusters, col_clusters


def parse(text_boxes, text_recs, output_path, strict=True):
    row_box_containment, col_box_containment = calculate_table_box_containment(text_boxes)
    if strict:
        row_clusters, col_clusters = strict_clarify(text_boxes, row_box_containment, col_box_containment)
    else:
        best = run_genetic_algorithm(text_boxes, row_box_containment, col_box_containment)
        row_clusters = cluster_boxes(row_box_containment, best.genes[0])
        col_clusters = cluster_boxes(col_box_containment, best.genes[1])
    table = create_table(text_boxes, text_recs, row_clusters, col_clusters)
    table.to_excel(output_path, index=False, header=False)
