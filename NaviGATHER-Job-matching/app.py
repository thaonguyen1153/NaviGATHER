from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_json():
    # Get JSON data from request
    data = request.get_json()

    # Check if data was received
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Example processing (e.g., echoing data with a new key)
    processed_data = {
        "original_data": data,
        "message": "Data received and processed successfully!",
        "data_length": len(data)  # Example of processing (e.g., counting items)
    }

    # Respond with the processed data as JSON
    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(debug=True)
