import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def read_csv(filepath):
    """
    Read a CSV file and return its contents as a pandas DataFrame.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: The CSV data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file can't be parsed as CSV
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
            
        return df
        
    except Exception as e:
        logger.error(f"Error reading CSV file {filepath}: {str(e)}")
        raise

def get_preview(filepath, n_rows=5):
    """
    Get a preview of the CSV file contents as an HTML table.
    
    Args:
        filepath (str): Path to the CSV file
        n_rows (int): Number of rows to preview
        
    Returns:
        str: HTML table string of the preview data
    """
    try:
        df = read_csv(filepath)
        preview_df = df.head(n_rows)
        
        # Convert DataFrame to HTML with custom styling
        html_table = preview_df.to_html(
            classes='preview-table',
            index=True,
            border=0,
            justify='left',
            escape=False,
            render_links=True,
            table_id='preview-table'
        )
        
        return html_table
    except Exception as e:
        logger.error(f"Error getting preview of {filepath}: {str(e)}")
        raise

def get_column_info(filepath):
    """
    Get information about the columns in the CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        dict: Information about each column including:
            - data type
            - number of non-null values
            - number of unique values
    """
    try:
        df = read_csv(filepath)
        info = {}
        
        for column in df.columns:
            info[column] = {
                'dtype': str(df[column].dtype),
                'non_null': df[column].count(),
                'unique': df[column].nunique()
            }
            
        return info
    except Exception as e:
        logger.error(f"Error getting column info for {filepath}: {str(e)}")
        raise

def standardize_text_column(filepath):
    """
    Standardize the text column in a CSV file by:
    1. Looking for a column named 'text' or 'reviews'
    2. If found, rename 'reviews' to 'text' if necessary
    3. If neither found, use the first text column and rename it to 'text'
    4. Keep only the text column in the output file
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        str: Path to the standardized file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Check for 'text' or 'reviews' column
        if 'text' in df.columns:
            # Already has 'text' column, but we'll still standardize
            text_column = 'text'
        elif 'reviews' in df.columns:
            # Rename 'reviews' to 'text'
            df = df.rename(columns={'reviews': 'text'})
            text_column = 'text'
        else:
            # Find the first text column
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) == 0:
                raise ValueError("No text columns found in the CSV file")
            
            # Rename the first text column to 'text'
            df = df.rename(columns={text_columns[0]: 'text'})
            text_column = 'text'
        
        # Keep only the text column
        df = df[['text']]
        
        # Save the standardized file
        output_path = filepath  # Overwrite the original file
        df.to_csv(output_path, index=False)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error standardizing text column in {filepath}: {str(e)}")
        raise

