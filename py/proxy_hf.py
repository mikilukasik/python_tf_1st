# flask endpoint to serve dataset from huggingface
import flask
from threading import Thread
from utils.helpers.get_dataset_from_hf import ChessDataset

chess_dataset = ChessDataset()


app = flask.Flask(__name__)


@app.route('/dataset', methods=['GET'])
def get_dataset():
    # query might have limit, default it to 100000
    limit = flask.request.args.get('limit')
    if limit:
        limit = int(limit)
    else:
        limit = 100000

    return chess_dataset.get_dataset_as_csv(limit)


def run_app():
    app.run(port=3910, debug=True, use_reloader=False, host='0.0.0.0')


# Start Flask app in a separate thread
flask_thread = Thread(target=run_app)
flask_thread.start()
