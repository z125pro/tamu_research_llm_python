import pandas as pd
import numpy as np

def csv_to_student_response_block(file_path):
    try:
        df = pd.read_csv(file_path, header=None, encoding="utf-8-sig")

        headers = pd.to_numeric(df.iloc[0], errors='coerce').fillna(0)
        max_points = pd.to_numeric(df.iloc[1], errors='coerce').fillna(0)

        students = df.iloc[2:]
        valid_students = students[pd.to_numeric(students[0], errors='coerce') > 0].copy()

        if valid_students.empty:
            return ""

        scores = valid_students.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(-1)
        answer_key = max_points[1:].values
        
        is_correct = np.abs(scores.values - answer_key) < 0.001
        tf_df = pd.DataFrame(np.where(is_correct, "T", "F"), index=valid_students.index)


        result_series = tf_df[0]

        for i in range(1, tf_df.shape[1]):
            sep = "||" if headers[i + 1] > 0 else " "
            result_series = result_series + sep + tf_df[i]

        return "\n".join(result_series)

    except Exception as e:
        return f"Error: {str(e)}"
    
# if __name__ == "__main__":
#     output_content = csv_to_student_response_block("llm_data/exam1/exam1_crawford.csv")
    
#     # 2. Define the output filename
#     output_filename = "student_response_crawford.txt"
    
#     # 3. Write to the file
#     if not output_content.startswith("Error:"):
#         with open(output_filename, "w", encoding="utf-8") as f:
#             f.write(output_content)
#         print(f"Success! Data written to {output_filename}")
#     else:
#         print(output_content)