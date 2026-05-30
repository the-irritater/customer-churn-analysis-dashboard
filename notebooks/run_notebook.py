import json
import sys
import io
import traceback
from pathlib import Path

def main():
    notebook_path = Path(__file__).parent / "Project.ipynb"
    log_path = Path(__file__).parent / "run_notebook.log"
    
    print(f"Reading notebook from {notebook_path}...")
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    namespace = {}
    
    # Define a helper print function that writes to both terminal and a string capture
    print("Starting execution of notebook cells...")
    
    code_cell_idx = 1
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if not source.strip():
                continue
            
            print(f"Running Cell {idx} (Code Cell {code_cell_idx})...")
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            error_occurred = False
            error_info = None
            
            try:
                # Execute the cell code in the shared namespace
                exec(source, namespace)
            except Exception as e:
                error_occurred = True
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                error_info = {
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": tb_lines
                }
            finally:
                # Restore original stdout/stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                
            # Get captured text
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            
            # Print to log
            if stdout_text:
                print(f"--- Cell {idx} Stdout ---")
                print(stdout_text.rstrip())
            if stderr_text:
                print(f"--- Cell {idx} Stderr ---")
                print(stderr_text.rstrip())
            if error_occurred:
                print(f"--- Cell {idx} Error ---")
                print(''.join(error_info["traceback"]).rstrip())
                
            # Prepare outputs list for the notebook cell
            outputs = []
            if stdout_text:
                outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [line + '\n' for line in stdout_text.splitlines()]
                })
            if stderr_text:
                outputs.append({
                    "output_type": "stream",
                    "name": "stderr",
                    "text": [line + '\n' for line in stderr_text.splitlines()]
                })
            if error_occurred:
                outputs.append(error_info)
                
            # Update cell in-place
            cell['outputs'] = outputs
            cell['execution_count'] = code_cell_idx
            code_cell_idx += 1
            
            if error_occurred:
                print(f"Aborting execution due to error in cell {idx}.")
                break
                
    # Save the executed notebook back
    print(f"Saving executed notebook to {notebook_path}...")
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook executed and saved successfully.")

if __name__ == '__main__':
    main()
