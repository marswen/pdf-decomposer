"""
https://developer.adobe.com/document-services/apis/pdf-extract/

https://github.com/adobe/pdfservices-python-sdk-samples/tree/main/src/extractpdf/extract_txt_table_info_with_figure_tables_rendition_from_pdf.py

```
pip install pdfservices-sdk==2.3.0

export ADOBE_CLIENT_ID=<YOUR CLIENT ID>
export ADOBE_CLIENT_SECRET=<YOUR CLIENT SECRET>
```
"""

import os
import shutil
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import \
    ExtractRenditionsElementType


class AdobePDFDecomposer:

    def __init__(self):
        pass

    @staticmethod
    def _save_result(result: FileRef, destination_file_path):
        if result._is_operation_result():
            result._logger.info(
                "Moving file at {tmp_file_path} to target {target_path}".format(tmp_file_path=result._file_path,
                                                                                target_path=destination_file_path))
            abs_path = os.path.abspath(destination_file_path)
            dir = os.path.dirname(abs_path)
            if not os.path.exists(dir):
                os.mkdir(dir)
            if not os.path.exists(abs_path):
                shutil.move(result._file_path, abs_path)
                return
            raise ("Output file {file} exists".format(file=destination_file_path))
        else:
            result._logger.error(
                "Invalid use of save_as(). Method invoked on FileRef instance which does not point to an operation "
                "result")
            raise AttributeError("Method save_as only allowed on operation results")

    def __call__(self, file_path, output_dir):
        # Initial setup, create credentials instance.
        api_id = os.getenv('ADOBE_CLIENT_ID')
        api_secret = os.getenv('ADOBE_CLIENT_SECRET')
        assert api_id is not None, \
            "Please set your Adobe CLIENT ID using environment variable (ADOBE_CLIENT_ID)"
        assert api_secret is not None, \
            "Please set your Adobe CLIENT SECRET using environment variable (ADOBE_CLIENT_SECRET)"
        credentials = Credentials.service_principal_credentials_builder(). \
            with_client_id(api_id). \
            with_client_secret(api_secret). \
            build()

        # Create an ExecutionContext using credentials and create a new operation instance.
        execution_context = ExecutionContext.create(credentials)
        extract_pdf_operation = ExtractPDFOperation.create_new()

        # Set operation input from a source file.
        source = FileRef.create_from_local_file(file_path)
        extract_pdf_operation.set_input(source)

        # Build ExtractPDF options and set them into the operation
        extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
            .with_elements_to_extract([ExtractElementType.TEXT, ExtractElementType.TABLES]) \
            .with_elements_to_extract_renditions([ExtractRenditionsElementType.TABLES,
                                                  ExtractRenditionsElementType.FIGURES]) \
            .build()
        extract_pdf_operation.set_options(extract_pdf_options)

        # Execute the operation.
        result: FileRef = extract_pdf_operation.execute(execution_context)

        # Save the result to the specified location.
        basename = os.path.basename(file_path)
        result_path = os.path.join(output_dir, basename[:basename.rindex('.')] + '.zip')
        self._save_result(result, result_path)
