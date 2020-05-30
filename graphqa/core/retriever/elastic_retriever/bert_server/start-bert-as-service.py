from bert_serving.server import BertServer
from bert_serving.server.helper import get_run_args


def main(args):
    server = BertServer(args=args)
    server.start()
    server.join()


if __name__ == '__main__':
    arguments = get_run_args()
    main(arguments)
