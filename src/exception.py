import sys

def error_message_details(error, error_detail: sys):
    '''
    Function to create custom exception handling message.
    Docs available online for "sys" library.
    '''
    _, _, exc_tb = error_detail.exc_info()
    
    filename = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno

    error_message = f'Error occured in Python script [{filename}] line number [{line_num}] error message [{error}]'

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_details(error=error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message