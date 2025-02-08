import streamlit as st
import boto3
import pandas as pd
import json
import base64
import numpy as np
import re

# Helper function to clean and format Python code from Claude's response
def clean_python_code(code_response):
    # Find Python code block between triple backticks
    code_match = re.search(r'```python\n(.*?)```', code_response, re.DOTALL)
    
    if code_match:
        # Extract code from the matched group
        code = code_match.group(1)
    else:
        # If no code block found, use the original cleaning logic
        code = code_response.replace("```python", "").replace("```", "").strip()
    
    # Remove any leading/trailing quotes
    code = code.strip('"\'')
    
    # Ensure proper line endings
    code = code.replace('\\n', '\n')
    
    return code

st.title("AI-Powered Data Transformation Bot")

# Initialize AWS Bedrock client
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id="ASIAVQIIYHBQNI2PANOB",
        aws_secret_access_key="bTfReadDMQs70pbq+d6EIDf26NTe6LqOysz0Ih89",
        aws_session_token="IQoJb3JpZ2luX2VjEHgaCXVzLWVhc3QtMSJFMEMCIAuC69lSLO8mXsXhFcy9UHoDxcUO9e4XnhNxOjuwwgQCAh9Emih1ibDUheX9I4q+ABdr1sc7x2cHXfT/Gc/PrqERKpsDCJD//////////wEQABoMMzc4NTEyMzU3NDcyIgyGUR9LxqdXVVKPAH0q7wJWxErr2M/rjGQeJBYyaTqi7rBBXQJoFd49aT+C+7INfsO1OFLbV5h0QQwIXV8DRjR8ZKDi2PwIWpeCnEet9w3D+d+1mSJL6ELy2V6I4176MENmuzJZkDIxrsCv5z/pM/hXWkuC88rlTXnXeVGxt0SVNwIVXEBkEh/pN8R5S8vYKU21VVgzlFl819h7T7nplK/Pgb2mRR9BeXIDemLvPStA8iBIp/TJPqR4rROd7n5HnhEjg1K1KIPH+ABNV7ahVf60scI3kJox2kjtbtfnegtKUOvgqoR26AkNiGtdLkiPcyhWW/R7rugygEzJM0mFspOROjvwCsYdewM8pD6WBZTXuXSM3b/lKOawYRpGLwd9qWOT+aBT6AEikwZ4NwuSdzVSAmwaZ1Myczx4Ef01XwN0HO8FCI7iWYdzBVO4ZzsQvKdkJPWDsGgjYok0+ABPZDaRrSO3eaBwNzkY2JGLGNeyWvv+m8/liI1BdJwxljZsMJjrnb0GOqgBbx/KyuKn9aUQ7V5YbZ2ArMJsd5bzz0dZGMUARP+GYKJh5IlOzI5nU5KyqvEiypCnSP4shGp1+eTO/RdMpV4zFXtERtcfDuvGngfp7jUnXEbppNjBeHvn1lHUmJenGDmvC6bgxJ9A9qvlZOU9Bw0i+N/0EV+OcfKPfCfV6q7d9cyJiAKb55gcMAFSWRj+a8C0HGC+ExqBZPuLqJBH1FSSgoRgW8wTJWD8",
        region_name="us-west-2"
    )

# Helper function to invoke Claude through Bedrock
def invoke_claude(prompt):
    client = get_bedrock_client()
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200000,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    
    response = client.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
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

st.subheader("Upload Files")

# First handle input file
input_file = st.file_uploader("Choose input CSV file", type="csv", key="input")

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
            
            # Now handle output file
            output_sample = st.file_uploader("Choose sample output CSV file (to map with)", type="csv", key="output")
            
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
                        
                        st.subheader("Output Schema")
                        with st.expander("Show Schema Details"):
                            st.write("Data Types:", dtypes)
                        
                        # Add cleaning options
                        st.subheader("Data Cleaning Options")
                        handle_missing = st.checkbox("Handle missing values", value=False)
                        remove_duplicates = st.checkbox("Remove duplicate rows", value=False)
                        fix_datatypes = st.checkbox("Fix data types", value=False)
                        
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

Input DataFrame columns (use these exact names with no changes):
{df.columns.tolist()}

Input DataFrame first few rows:
{df.head().to_string()}

Data quality insights:
- Missing values by column: {missing_values[missing_values > 0].to_dict() if missing_values.any() else 'None'}
- Duplicate rows: {duplicates}

Selected cleaning steps:
{chr(10).join(cleaning_steps)}

Special instructions:
{special_instructions if special_instructions else 'None'}

IMPORTANT: 
- Use ONLY the 'df' variable that is already loaded in memory
- Use the EXACT column names as shown above
- Return ONLY valid Python code that creates a cleaned_df variable
- Always check column existence before operations
- Handle each column separately with proper error handling
- Do not try to read or write any files
- Do not include any explanations or markdown formatting
- Do not fill missing values with placeholder text like 'Not Specified', 'None', 'Unknown', etc
- Only fill missing values when you can determine the correct value with high confidence
- Leave missing values as-is if the correct value cannot be determined

Example format:
cleaned_df = df.copy()

# Handle missing values only when correct value is known
if 'Column1' in cleaned_df.columns:
    # Only fill if we can determine the right value
    cleaned_df['Column1'] = cleaned_df['Column1'].fillna(method='ffill')  # Example: forward fill if values are sequential
    
if 'Column2' in cleaned_df.columns:
    # Only fill numeric columns with statistical measures if appropriate
    cleaned_df['Column2'] = cleaned_df['Column2'].fillna(cleaned_df['Column2'].median())

# Remove duplicates if needed
cleaned_df = cleaned_df.drop_duplicates()
"""

                            with st.spinner("Analyzing data and generating cleaning suggestions..."):
                                cleaning_code = invoke_claude(cleaning_prompt)
                                cleaning_code = clean_python_code(cleaning_code)
                                
                                # Add a new prompt to get cleaning summary
                                summary_prompt = f"""Analyze this Python cleaning code and provide a brief, bullet-point summary of the cleaning steps being performed. Focus on what the code does, not how it does it.

Code to analyze:
{cleaning_code}

Format your response as a simple bullet list in short sentences with line breaks between each bullet point. Do not include any additional text or markdown."""

                                # Get summary from Claude
                                cleaning_summary = invoke_claude(summary_prompt)

                                # Display summary and add expander for code
                                st.subheader("Data Cleaning Steps")
                                st.write(cleaning_summary)

                                # Add expander to show code
                                with st.expander("Show Python Code"):
                                    st.code(cleaning_code, language="python")

                                try:
                                    # Execute cleaning code
                                    local_namespace = {'df': df.copy(), 'pd': pd}
                                    exec(cleaning_code, globals(), local_namespace)
                                    cleaned_df = local_namespace.get('cleaned_df')
                                    
                                    if cleaned_df is not None:
                                        st.subheader("Cleaned Data Preview")
                                        st.dataframe(cleaned_df.head())
                                        
                                        # Store cleaned_df in session state
                                        st.session_state['cleaned_df'] = cleaned_df
                                        
                                        # Add field mapping section with improved UI
                                        st.subheader("AI-Generated Field Mappings")
                                        with st.expander("Review and modify field mappings"):
                                            # Create mapping rows
                                            input_fields = cleaned_df.columns.tolist()
                                            output_fields = output_df.columns.tolist()
                                            
                                            # Create a container for better spacing
                                            mapping_container = st.container()
                                            
                                            with mapping_container:
                                                # Headers row
                                                col1, col2 = st.columns([1, 1])
                                                with col1:
                                                    st.markdown("**Input Field**")
                                                with col2:
                                                    st.markdown("**Output Field**")
                                                
                                                for output_field in output_fields:
                                                    col1, col2 = st.columns([1, 1])
                                                    
                                                    with col1:
                                                        default_value = output_field if output_field in input_fields else ""
                                                        selected_input = st.selectbox(
                                                            label=f"Map to {output_field}",
                                                            options=[""] + input_fields,
                                                            index=input_fields.index(default_value) + 1 if default_value in input_fields else 0,
                                                            label_visibility="collapsed",
                                                            key=f"mapping_{output_field}"
                                                        )
                                                    
                                                    with col2:
                                                        st.text(output_field)
                                            
                                            st.info("Select the input field to map to each output field. Leave blank to skip mapping.")
                                        
                                        # Get the user-defined mappings
                                        field_mappings = {
                                            output_field: st.session_state[f"mapping_{output_field}"]
                                            for output_field in output_fields
                                            if st.session_state[f"mapping_{output_field}"] != ""
                                        }
                                        
                                        # Update transform prompt with user-defined mappings
                                        transform_prompt = f"""You are a Python data transformation expert. Write code to transform the cleaned dataframe to match the output format using these specific field mappings:

{json.dumps(field_mappings, indent=2)}

INPUT SCHEMA:
{pd.DataFrame({
    'Column': cleaned_df.columns.tolist(),
    'CurrentType': cleaned_df.dtypes.astype(str).tolist()
}).to_string(index=False)}

OUTPUT SCHEMA:
{pd.DataFrame({
    'Column': output_df.columns.tolist(),
    'RequiredType': [dtypes[col] for col in output_df.columns]
}).to_string(index=False)}

Sample input rows:
{cleaned_df.head(30).to_string(index=False)}

Sample output format:
{output_df.head(30).to_string(index=False)}

Generate Python code that:
1. Creates transformed_df from cleaned_df
2. Uses the provided field mappings to map input columns to output columns
3. Converts data types safely
4. Handles missing data appropriately
5. IMPORTANT: Include the actual function call at the end to create transformed_df

Use this structure:
def transform_dataframe(cleaned_df):
    transformed_df = pd.DataFrame(index=cleaned_df.index)
    # ... transformation logic using the provided mappings ...
    return transformed_df

# Actually call the function to create transformed_df
transformed_df = transform_dataframe(cleaned_df)
"""

                                        with st.spinner("Generating transformation code..."):
                                            transform_code = invoke_claude(transform_prompt)
                                            transform_code = clean_python_code(transform_code)
                                            
                                            # Add a new prompt to get transformation summary
                                            summary_prompt = f"""Analyze this Python transformation code and provide a brief, bullet-point summary of the transformation steps being performed. Focus on what the code does, not how it does it.

Code to analyze:
{transform_code}

Format your response as a simple bullet list in short sentences with line breaks between each bullet point. Do not include any additional text or markdown."""

                                            # Get summary from Claude
                                            transform_summary = invoke_claude(summary_prompt)

                                            # Display summary and add expander for code
                                            st.subheader("Transformation Steps")
                                            st.write(transform_summary)

                                            # Add expander to show code
                                            with st.expander("Show Python Code"):
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
                                                        
                                                        # Add chat interface for additional transformations
                                                        st.subheader("Need Additional Transformations?")
                                                        user_request = st.text_area(
                                                            "Describe any additional transformations needed:",
                                                            placeholder="Example: Convert all text to uppercase, calculate new columns, etc."
                                                        )
                                                        
                                                        if st.button("Process Additional Transformation"):
                                                            if user_request:
                                                                with st.spinner("Generating additional transformation..."):
                                                                    # Generate transformation code based on user request
                                                                    additional_prompt = f"""You are a Python data transformation expert. The user needs additional transformation on their DataFrame.

Current DataFrame columns and sample data:
{transformed_df.head().to_string()}

User request: {user_request}

Provide Python code to perform this transformation. The code should:
1. Use the existing transformed_df
2. Create a new DataFrame with the requested changes
3. Include the actual function call
4. Handle errors appropriately
5. Return the modified DataFrame

IMPORTANT: Return ONLY the Python code without any explanations."""

                                                                    additional_code = invoke_claude(additional_prompt)
                                                                    additional_code = clean_python_code(additional_code)
                                                                    
                                                                    # Get a summary of changes
                                                                    summary_prompt = f"""Analyze this Python transformation code and provide a brief, clear explanation of what changes it will make to the data.

Code to analyze:
{additional_code}

Provide a simple, non-technical explanation."""
                                                                    
                                                                    changes_summary = invoke_claude(summary_prompt)
                                                                    
                                                                    # Show summary and code
                                                                    st.write("Here's what this transformation will do:")
                                                                    st.write(changes_summary)
                                                                    
                                                                    with st.expander("Show Python Code"):
                                                                        st.code(additional_code, language="python")
                                                                    
                                                                    if st.button("Apply Additional Transformation"):
                                                                        try:
                                                                            # Execute the transformation
                                                                            local_namespace = {
                                                                                'transformed_df': transformed_df.copy(),
                                                                                'pd': pd,
                                                                                'np': np
                                                                            }
                                                                            exec(additional_code, globals(), local_namespace)
                                                                            new_df = local_namespace.get('transformed_df')
                                                                            
                                                                            if new_df is not None:
                                                                                st.write("Preview of additionally transformed data:")
                                                                                st.dataframe(new_df.head())
                                                                                
                                                                                # Add download button for the latest version
                                                                                st.download_button(
                                                                                    label="Download additionally transformed data",
                                                                                    data=new_df.to_csv(index=False),
                                                                                    file_name="additionally_transformed_data.csv",
                                                                                    mime="text/csv"
                                                                                )
                                                                            else:
                                                                                st.error("Additional transformation did not produce expected output")
                                                                        except Exception as e:
                                                                            st.error(f"Error during additional transformation: {str(e)}")
                                                            else:
                                                                st.warning("Please describe the additional transformation needed")
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