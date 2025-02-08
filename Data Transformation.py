# //data transformation chatbot

# //1. data cleaning
# //2. data integration
# //3. data transformation
# //4. data loading

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import boto3
import io

class DataTransformationBot:
    def __init__(self):
        self.input_df: Optional[pd.DataFrame] = None
        self.desired_schema: Optional[Dict] = None
        self.transformation_rules: Dict = {}
        
    def analyze_input_file(self, file_path: str, sample_rows: int = 5) -> Dict:
        """Analyze input file structure and data samples"""
        self.input_df = pd.read_csv(file_path)
        
        analysis = {
            'headers': list(self.input_df.columns),
            'dtypes': self.input_df.dtypes.to_dict(),
            'sample_data': self.input_df.head(sample_rows).to_dict('records'),
            'total_rows': len(self.input_df)
        }
        return analysis
    
    def set_desired_schema(self, schema: Dict):
        """Set the desired output schema"""
        self.desired_schema = schema
        
    def suggest_transformations(self) -> Dict[str, List[str]]:
        """Suggest possible transformations based on input and desired schema"""
        suggestions = {}
        
        for desired_col in self.desired_schema['columns']:
            matching_cols = self._find_matching_columns(desired_col)
            suggestions[desired_col] = matching_cols
            
        return suggestions
    
    def _find_matching_columns(self, desired_col: str) -> List[str]:
        """Find potential matching columns from input data"""
        if self.input_df is None:
            return []
            
        matches = []
        for col in self.input_df.columns:
            # Add simple string matching logic
            if col.lower().replace('_', '') == desired_col.lower().replace('_', ''):
                matches.append(col)
            # Add fuzzy matching logic here if needed
        return matches
    
    def set_transformation_rule(self, target_column: str, transformation_code: str):
        """Set transformation rule for a specific column"""
        self.transformation_rules[target_column] = transformation_code
        
    def transform_data(self) -> pd.DataFrame:
        """Apply transformations and return transformed DataFrame"""
        if self.input_df is None:
            raise ValueError("Input data not loaded")
            
        result_df = pd.DataFrame()
        
        for col, transform_rule in self.transformation_rules.items():
            try:
                # Safely execute transformation rule
                result_df[col] = eval(f"self.input_df.{transform_rule}")
            except Exception as e:
                print(f"Error transforming column {col}: {str(e)}")
                
        return result_df
    
    def export_to_csv(self, output_path: str, transformed_df: Optional[pd.DataFrame] = None):
        """Export transformed data to CSV"""
        if transformed_df is None:
            transformed_df = self.transform_data()
        transformed_df.to_csv(output_path, index=False)

# AWS Lambda handler
def lambda_handler(event, context):
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Get input file from S3
        input_bucket = event['input_bucket']
        input_key = event['input_key']
        output_bucket = event['output_bucket']
        output_key = event['output_key']
        transformation_rules = event['transformation_rules']
        desired_schema = event['desired_schema']
        
        # Read input file from S3
        response = s3.get_object(Bucket=input_bucket, Key=input_key)
        input_data = response['Body'].read()
        
        # Create DataFrame from input
        input_df = pd.read_csv(io.BytesIO(input_data))
        
        # Initialize bot and process transformation
        bot = DataTransformationBot()
        bot.input_df = input_df
        bot.set_desired_schema(desired_schema)
        
        # Set transformation rules
        for column, rule in transformation_rules.items():
            bot.set_transformation_rule(column, rule)
        
        # Transform data
        transformed_df = bot.transform_data()
        
        # Save to S3
        csv_buffer = io.StringIO()
        transformed_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=csv_buffer.getvalue()
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Transformation completed successfully',
                'output_location': f's3://{output_bucket}/{output_key}'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

# Local development usage example
if __name__ == "__main__":
    # Initialize bot
    bot = DataTransformationBot()
    
    # Analyze input file
    input_analysis = bot.analyze_input_file('input.csv')
    print("Input File Analysis:", json.dumps(input_analysis, indent=2))
    
    # Set desired schema
    desired_schema = {
        'columns': ['transformed_col1', 'transformed_col2'],
        'dtypes': {'transformed_col1': 'string', 'transformed_col2': 'float'}
    }
    bot.set_desired_schema(desired_schema)
    
    # Get transformation suggestions
    suggestions = bot.suggest_transformations()
    print("Transformation Suggestions:", json.dumps(suggestions, indent=2))
    
    # Set transformation rules
    bot.set_transformation_rule('transformed_col1', "column1.str.upper()")
    bot.set_transformation_rule('transformed_col2', "column2 * 2")
    
    # Transform and export
    transformed_df = bot.transform_data()
    bot.export_to_csv('output.csv', transformed_df)
