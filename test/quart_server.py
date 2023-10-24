from quart import Quart, request, jsonify

app = Quart(__name__)

@app.route('/yolo', methods=['POST'])
async def process_yolo():
    print("Received request")
    files = await request.files
    form = await request.form
    print(form)
    print(files)
    image_data = files.get('image')
    json_str = form.get('json_data')
    print(json_str)
    return "Received request"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=50048)