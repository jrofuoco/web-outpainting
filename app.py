from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from gradio_client import Client, handle_file
import tempfile
import os
import shutil
import logging

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Gradio client
client = Client("multimodalart/flux-fill-outpaint")

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/outpaint', methods=['POST'])
def outpaint():
    input_tmp = None
    output_tmp = None
    temp_dir = None
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        image = request.files['image']
        if image.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Create a temporary directory that Flask can access
        temp_dir = tempfile.mkdtemp()
        logging.debug(f"Created temporary directory: {temp_dir}")
        
        # Save the input image
        input_tmp = os.path.join(temp_dir, "input.png")
        image.save(input_tmp)
        logging.debug(f"Saved input image to: {input_tmp}")
        
        # Run Gradio client
        try:
            # Step 1: Upload the image
            client.predict(
                handle_file(input_tmp),
                api_name="/use_output_as_input"
            )
            
            # Step 2: Run the inpaint operation
            result = client.predict(
                image=handle_file(input_tmp),
                width=720,
                height=1280,
                overlap_percentage=10,
                num_inference_steps=28,
                resize_option="75%",
                custom_resize_percentage=50,
                prompt_input="",
                alignment="Middle",
                overlap_left=True,
                overlap_right=True,
                overlap_top=True,
                overlap_bottom=True,
                api_name="/inpaint"
            )
            
            logging.debug(f"Result type: {type(result)}")
            logging.debug(f"Result content: {result}")
        except Exception as e:
            logging.exception(f"Error in Gradio client: {str(e)}")
            return jsonify({"error": f"Error in image processing: {str(e)}"}), 500
        
        # Extract the result file path
        outpaint_result = None
        if isinstance(result, str):
            outpaint_result = result
        elif isinstance(result, list) and result:
            if isinstance(result[0], dict) and "path" in result[0]:
                outpaint_result = result[0]["path"]
            else:
                outpaint_result = str(result[0])
        elif isinstance(result, dict) and "path" in result:
            outpaint_result = result["path"]
        elif isinstance(result, tuple) and len(result) > 0:
            outpaint_result = result[0]  # Take the first item in the tuple
        else:
            raise ValueError(f"Unexpected result format: {result}")
        
        if not outpaint_result or not os.path.exists(outpaint_result):
            raise ValueError(f"Result file not found: {outpaint_result}")
        
        # Copy the result to our temp directory
        output_tmp = os.path.join(temp_dir, "output.webp")
        shutil.copy2(outpaint_result, output_tmp)
        logging.debug(f"Copied result to: {output_tmp}")
        
        # Check if the copied file exists
        if not os.path.exists(output_tmp):
            raise ValueError(f"Copied file not found: {output_tmp}")
        
        # Return the file from our temp directory
        return send_file(
            output_tmp, 
            mimetype='image/webp',
            as_attachment=True,
            download_name='outpainted-image.webp'
        )
            
    except Exception as e:
        logging.exception(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logging.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logging.warning(f"Failed to delete temporary directory: {str(e)}")
    
if __name__ == '__main__':
    # Make sure templates directory exists
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)