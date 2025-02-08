import streamlit as st
import boto3
import pandas as pd
import json
import base64
import numpy as np

st.title("AI-Powered Data Transformation Bot")

# Initialize AWS Bedrock client
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id="ASIAVQIIYHBQNI2PANOB",
        aws_secret_access_key="bTfReadDMQs70pbq+d6EIDf26NTe6LqOysz0Ih89",
        aws_session_token="IQoJb3JpZ2luX2VjEHgaCXVzLWVhc3QtMSJFMEMCIAuC69lSLO8mXsXhFcy9UHoDxcUO9e4XnhNxOjuwwgQCAh9Emih1ibDUheX9I4q+ABdr1sc7x2cHXfT/Gc/PrqERKpsDCJD//////////wEQABoMMzc4NTEyMzU3NDcyIgyGUR9LxqdXVVKPAH0q7wJWxErr2M/rjGQeJBYyaTqi7rBBXQJoFd49aT+C+7INfsO1OFLbV5h0QQwIXV8DRjR8ZKDi2PwIWpeCnEet9w3D+d+1mSJL6ELy2V6I4176MENmuzJZkDIxrsCv5z/pM/hXWkuC88rlTXnXeVGxt0SVNwIVXEBkEh/pN8R5S8vYKU21VVgzlFl819h7T7nplK/Pgb2mRR9BeXIDemLvPStA8iBIp/TJPqR4rROd7n5HnhEjg1K1KIPH+ABNV7ahVf60scI3kJox2kjtbtfnegtKUOvgqoR26AkNiGtdLkiPcyhWW/R7rugygEzJM0mFspOROjvwCsYdewM8pD6WBZTXuXSM3b/lKOawYRpGLwd9qWOT+aBT6AEikwZ4NwuSdzVSAmwaZ1Myczx4Ef01XwN0HO8FCI7iWYdzBVO4ZzsQvKdkJPWDsGgjYok0+ABPZDaRrSO3eaBwNzkY2JGLGNeyWvv+m8/liI1BdJwxljZsMJjrnb0GOqgBbx/KyuKn9aUQ7V5YbZ2ArMJsd5bzz0dZGMUARP+GYKJh5IlOzI5nU5KyqvEiypCnSP4shGp1+eTO/RdMpV4zFXtERtcfDuvGngfp7jUnXEbppNjBeHvn1lHUmJenGDmvC6bgxJ9A9qvlZOU9Bw0i+N/0EV+OcfKPfCfV6q7d9cyJiAKb55gcMAFSWRj+a8C0HGC+ExqBZPuLqJBH1FSSgoRgW8wTJWD8",
        region_name="us-east-1"
    )

# Helper function to invoke Claude through Bedrock
def invoke_claude(prompt):
    client = get_bedrock_client()
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    
    response = client.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=body
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

# Helper function to get Claude's suggestions
def get_claude_suggestions(df_head, desired_schema):
    prompt = f"""You are a data transformation expert. I need help transforming data.

Given this input dataframe (showing first few rows):
{df_head.to_string()}

And the desired output schema:
{json.dumps(desired_schema, indent=2)}

Please suggest transformation rules for each output column. Your response must be ONLY a valid JSON object with column names as keys and transformation rules as values. 
For example:
{{
    "column1": "transformation rule 1",
    "column2": "transformation rule 2"
}}"""

    try:
        response_text = invoke_claude(prompt)
        
        # Try to find JSON content within the response
        try:
            # First attempt: try to parse the entire response
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Second attempt: try to find JSON-like content between curly braces
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                st.error("Could not parse AI response into JSON format")
                st.text("Raw AI response:")
                st.text(response_text)
                return {}
                
    except Exception as e:
        st.error(f"Error in AI response: {str(e)}")
        st.text("Raw AI response:")
        st.text(response_text)
        return {}

# File upload for input and output samples
st.subheader("Upload Files")
col1, col2 = st.columns(2)

with col1:
    input_file = st.file_uploader("Choose input CSV file", type="csv", key="input")
    
with col2:
    output_sample = st.file_uploader("Choose sample output CSV file", type="csv", key="output")

if input_file is not None:
    try:
        # Read the input file with more flexible parsing options
        df = pd.read_csv(
            input_file,
            on_bad_lines='warn',
            escapechar='\\',
            quoting=1,
            quotechar='"',
            encoding='utf-8-sig'
        )
        
        # Display input data preview
        st.subheader("Input Data Preview")
        if df.empty:
            st.error("The input file is empty")
        else:
            st.dataframe(df.head())
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Display data quality insights
            st.subheader("Data Quality Insights")
            missing_values = df.isnull().sum()
            if missing_values.any():
                st.write("Missing Values:")
                st.write(missing_values[missing_values > 0])
            
            # Display duplicate rows if any
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.write(f"Number of duplicate rows: {duplicates}")
        
        if output_sample is not None:
            try:
                output_df = pd.read_csv(
                    output_sample,
                    on_bad_lines='warn',
                    escapechar='\\',
                    quoting=1,
                    quotechar='"',
                    encoding='utf-8-sig'
                )
                if output_df.empty:
                    st.error("The output sample file is empty")
                else:
                    columns = output_df.columns.tolist()
                    dtypes = {col: "string" if pd.api.types.is_string_dtype(output_df[col])
                            else "float" if pd.api.types.is_float_dtype(output_df[col])
                            else "integer" if pd.api.types.is_integer_dtype(output_df[col])
                            else "string" for col in columns}
                    
                    st.subheader("Detected Output Schema")
                    st.write("Columns:", columns)
                    st.write("Data Types:", dtypes)
                    
                    # Add cleaning options
                    st.subheader("Data Cleaning Options")
                    handle_missing = st.checkbox("Handle missing values", value=True)
                    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
                    fix_datatypes = st.checkbox("Fix data types", value=True)
                    
                    # Add text area for special instructions
                    special_instructions = st.text_area(
                        "Special cleaning instructions (optional)",
                        placeholder="Enter any special instructions for data cleaning..."
                    )
                    
                    if st.button("Clean and Transform Data"):
                        # Modify cleaning prompt based on selected options
                        cleaning_steps = []
                        if handle_missing:
                            cleaning_steps.append("1. Handle missing values appropriately (fill or remove)")
                        if remove_duplicates:
                            cleaning_steps.append("2. Remove duplicate rows if present")
                        if fix_datatypes:
                            cleaning_steps.append("3. Fix data types if needed")
                        
                        cleaning_prompt = f"""You are a data cleaning expert. Analyze this input dataframe and provide Python code to clean it.

Input DataFrame (first few rows):
{df.head().to_string()}

Data quality insights:
- Missing values: {missing_values[missing_values > 0].to_dict() if missing_values.any() else 'None'}
- Duplicate rows: {duplicates}

Selected cleaning steps:
{chr(10).join(cleaning_steps)}

Special instructions:
{special_instructions if special_instructions else 'None'}

IMPORTANT: 
- Use ONLY the 'df' variable that is already loaded in memory
- Return ONLY valid Python code that creates a cleaned_df variable
- Do not try to read or write any files
- Do not include any explanations or markdown formatting

Example format:
cleaned_df = df.copy()
cleaned_df = cleaned_df.dropna()
cleaned_df = cleaned_df.drop_duplicates()
"""

                        with st.spinner("Analyzing data and generating cleaning suggestions..."):
                            cleaning_code = invoke_claude(cleaning_prompt)
                            st.subheader("Data Cleaning Steps")
                            st.code(cleaning_code, language="python")
                            
                            try:
                                # Execute cleaning code
                                local_namespace = {'df': df.copy(), 'pd': pd}
                                exec(cleaning_code, globals(), local_namespace)
                                cleaned_df = local_namespace.get('cleaned_df')
                                
                                if cleaned_df is not None:
                                    st.subheader("Cleaned Data Preview")
                                    st.dataframe(cleaned_df.head())
                                    
                                    # Now get transformation suggestions
                                    transform_prompt = f"""You are a Python data transformation expert. Write code to transform the cleaned dataframe to match the output format.

Input DataFrame:
{cleaned_df.head().to_string()}

Target Output Format:
{output_df.head().to_string()}

Requirements:
1. Create all required columns: {columns}
2. Match these data types: {dtypes}
3. Transform the data to match the output format exactly

IMPORTANT:
- Use the 'cleaned_df' variable as input
- Create a new 'transformed_df' variable
- Return ONLY valid Python code
- Do not include any explanations or markdown
- Do not use any external files
- Use only pandas operations

Example format:
transformed_df = cleaned_df.copy()
transformed_df['new_column'] = transformed_df['old_column'].apply(lambda x: x.upper())
"""

                                    with st.spinner("Generating transformation code..."):
                                        transform_code = invoke_claude(transform_prompt)
                                        transform_code = clean_python_code(transform_code)
                                        st.subheader("Transformation Steps")
                                        st.code(transform_code, language="python")
                                        
                                        try:
                                            # Execute transformation code with necessary imports
                                            local_namespace = {
                                                'cleaned_df': cleaned_df.copy(),
                                                'pd': pd,
                                                'np': np,
                                                'transformed_df': None
                                            }
                                            exec(transform_code, globals(), local_namespace)
                                            transformed_df = local_namespace.get('transformed_df')
                                            
                                            if transformed_df is not None:
                                                st.subheader("Transformed Data Preview")
                                                st.dataframe(transformed_df.head())
                                                
                                                # Validate transformed data
                                                schema_valid = True
                                                for col, dtype in dtypes.items():
                                                    if col not in transformed_df.columns:
                                                        st.error(f"Missing column: {col}")
                                                        schema_valid = False
                                                
                                                if schema_valid:
                                                    # Download buttons for both cleaned and transformed data
                                                    col1, col2 = st.columns(2)
                                                    with col1:
                                                        cleaned_csv = cleaned_df.to_csv(index=False)
                                                        st.download_button(
                                                            label="Download cleaned data",
                                                            data=cleaned_csv,
                                                            file_name="cleaned_data.csv",
                                                            mime="text/csv"
                                                        )
                                                    with col2:
                                                        transformed_csv = transformed_df.to_csv(index=False)
                                                        st.download_button(
                                                            label="Download transformed data",
                                                            data=transformed_csv,
                                                            file_name="transformed_data.csv",
                                                            mime="text/csv"
                                                        )
                                            else:
                                                st.error("Transformation did not produce expected output")
                                        except Exception as e:
                                            st.error(f"Error during transformation: {str(e)}")
                                else:
                                    st.error("Cleaning did not produce expected output")
                            except Exception as e:
                                st.error(f"Error during cleaning: {str(e)}")
                                
            except pd.errors.EmptyDataError:
                st.error("The output sample file is empty")
            except Exception as e:
                st.error(f"Error reading output sample file: {str(e)}")
    except pd.errors.EmptyDataError:
        st.error("The input file is empty")
    except Exception as e:
        st.error(f"Error reading input file: {str(e)}") 

# Helper function to clean and format Python code from Claude's response
def clean_python_code(code_response):
    # Remove any markdown code block indicators
    code = code_response.replace("```python", "").replace("```", "").strip()
    
    # Remove any leading/trailing quotes
    code = code.strip('"\'')
    
    # Ensure proper line endings
    code = code.replace('\\n', '\n')
    
    return code 