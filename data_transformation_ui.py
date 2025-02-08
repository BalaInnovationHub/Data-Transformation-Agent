import streamlit as st
import boto3
import pandas as pd
import json
import base64

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
            on_bad_lines='warn',  # Warn about problematic lines instead of failing
            escapechar='\\',      # Handle escaped characters
            quoting=1,            # Handle quoted fields (1 = QUOTE_ALL)
            quotechar='"',        # Specify quote character
            encoding='utf-8-sig'  # Handle BOM and different encodings
        )
        
        # Display input data preview and any parsing warnings
        st.subheader("Input Data Preview")
        if df.empty:
            st.error("The input file is empty")
        else:
            st.dataframe(df.head())
            
            # Display shape information
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Automatically define schema from output sample if provided
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
                    
                    # Automatically determine dtypes
                    dtypes = {}
                    for col in columns:
                        if pd.api.types.is_integer_dtype(output_df[col]):
                            dtypes[col] = "integer"
                        elif pd.api.types.is_float_dtype(output_df[col]):
                            dtypes[col] = "float"
                        else:
                            dtypes[col] = "string"
                    
                    # Display detected schema
                    st.subheader("Detected Output Schema")
                    st.write("Columns:", columns)
                    st.write("Data Types:", dtypes)
                    
                    if st.button("Get AI Suggestions"):
                        desired_schema = {
                            'columns': columns,
                            'dtypes': dtypes
                        }
                        
                        # Get transformation suggestions from Claude
                        with st.spinner("Getting AI suggestions..."):
                            try:
                                suggestions = get_claude_suggestions(df.head(), desired_schema)
                                st.subheader("AI Transformation Suggestions")
                                st.json(suggestions)
                            
                                # Input for transformation rules
                                st.subheader("Set Transformation Rules")
                                transformation_rules = {}
                                for col in columns:
                                    default_rule = suggestions.get(col, "")
                                    rule = st.text_area(f"Transformation rule for {col}", value=default_rule)
                                    if rule:
                                        transformation_rules[col] = rule
                                
                                if transformation_rules and st.button("Transform Data"):
                                    # Get Claude to help with the transformation
                                    transform_prompt = f"""Given this input dataframe (showing first few rows):
{df.head().to_string()}

And these transformation rules:
{json.dumps(transformation_rules, indent=2)}

Please provide Python code to transform the data according to these rules. The code should create a new dataframe called 'transformed_df'."""

                                    with st.spinner("Generating transformation code..."):
                                        transformation_code = invoke_claude(transform_prompt)
                                        try:
                                            # Create a local namespace for execution
                                            local_namespace = {'df': df.copy(), 'pd': pd}
                                            exec(transformation_code, globals(), local_namespace)
                                            
                                            # Get the transformed_df from the local namespace
                                            transformed_df = local_namespace.get('transformed_df')
                                            
                                            if transformed_df is not None:
                                                st.subheader("Transformed Data Preview")
                                                st.dataframe(transformed_df.head())
                                                
                                                # Validate transformed data against schema
                                                schema_valid = True
                                                for col, dtype in dtypes.items():
                                                    if col not in transformed_df.columns:
                                                        st.error(f"Missing column: {col}")
                                                        schema_valid = False
                                                
                                                if schema_valid:
                                                    # Download button for transformed data
                                                    csv = transformed_df.to_csv(index=False)
                                                    st.download_button(
                                                        label="Download transformed data",
                                                        data=csv,
                                                        file_name="transformed_data.csv",
                                                        mime="text/csv"
                                                    )
                                            else:
                                                st.error("Transformation did not produce expected output")
                                        except Exception as e:
                                            st.error(f"Error during transformation: {str(e)}")
                            except Exception as e:
                                st.error(f"Error getting AI suggestions: {str(e)}")
            except pd.errors.EmptyDataError:
                st.error("The output sample file is empty")
            except Exception as e:
                st.error(f"Error reading output sample file: {str(e)}")
    except pd.errors.EmptyDataError:
        st.error("The input file is empty")
    except Exception as e:
        st.error(f"Error reading input file: {str(e)}") 