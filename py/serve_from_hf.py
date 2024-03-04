import flask
# from flask_executor import Executor
from threading import Thread
from utils.helpers.get_dataset_from_hf_st import ChessDataset
import gzip
import io

app = flask.Flask(__name__)
# executor = Executor(app)
chess_dataset = ChessDataset()


@app.route('/dataset', methods=['GET'])
def get_dataset():
    limit = flask.request.args.get('limit', 100000)
    limit = int(limit)

    print("limit:", limit)

    csv_data = chess_dataset.get_dataset_as_csv(limit)

    # Compress the CSV data
    compressed_data = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_data, mode='wb') as gz:
        gz.write(csv_data.encode('utf-8'))

    compressed_data.seek(0)

    # Return the compressed data with appropriate headers
    response = flask.make_response(compressed_data.getvalue())
    # response.headers['Content-Disposition'] = 'attachment; filename=dataset.csv.gz'
    response.headers['Content-Encoding'] = 'gzip'
    # response.headers['Content-Type'] = 'application/octet-stream'
    return response


def run_app():
    app.run(port=3910, debug=True, use_reloader=False, host='0.0.0.0')


flask_thread = Thread(target=run_app)
flask_thread.start()
