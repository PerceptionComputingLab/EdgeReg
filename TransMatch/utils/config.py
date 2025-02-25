import argparse

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='4')
parser.add_argument("--train_data_file", type=str,
                    dest="train_dir", default='../autodl-tmp/OASIS_L2R/All/')
parser.add_argument("--test_data_file", type=str,
                    dest="test_dir", default='../autodl-tmp/OASIS_L2R/Test/')

parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')

# train时参数
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=200)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--edge_sim_loss", type=str, help="edge similarity loss: mse or ncc",
                    dest="edge_sim_loss", default="ncc")
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--beta", type=float, help="edge loss parameter",
                    dest="beta", default=1)
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=10)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='../autodl-tmp/Checkpoint/')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')
parser.add_argument("--input_channel", type=str, help="model input channel",
                    dest="input_channel", default=14)
args = parser.parse_args()
