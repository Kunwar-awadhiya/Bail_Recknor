from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Flask API!"


@app.route('/app', methods=['GET'])
def get_data():
    data = {
        'message': 'Hello, this is your data!',
        'status': 'success'
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
