
import torch
import argparse
import sys

from network import RnnFactory

class Setting:
    
    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=200, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
    
    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=1024, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
    
    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])
                
        parser = argparse.ArgumentParser()        
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        
        ### command line parameters ###
        # training
        parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')        
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default = 0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=50, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')        
        
        # data management
        parser.add_argument('--dataset', default='loc-gowalla_totalCheckins.txt', type=str, help='the dataset under ../../dataset/<dataset.txt> to load')        
        parser.add_argument('--sequence-length', default=20, type=int, help='amount of locations to process in one pass (batching)')        
        parser.add_argument('--min-checkins', default=101, type=int, help='amount of checkins required per user')
        
        # evaluation
        parser.add_argument('--validate-on-latest', default=False, const=True, nargs='?', type=bool, help='use only latest sequence sample to validate')
        parser.add_argument('--validate-epoch', default=5, type=int, help='run validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int, help='report every x user on evaluation')
        parser.add_argument('--skip-recall', default=False, const=True, nargs='?', type=bool, help='skip recall@1,5,10 (only evaluate MAP)')
                
        args = parser.parse_args()
        
        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s
        
        # data management
        self.dataset_file = '../../dataset/{}'.format(args.dataset)
        self.max_users = 0 # use all available users
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.min_checkins = args.min_checkins
        
        # evaluation
        self.validate_on_latest = args.validate_on_latest
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user
        self.validate_map_only = args.skip_recall
        self.validate_recall = not args.skip_recall
     
        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)        
    
    def __str__(self):        
        return ('parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n'\
            + 'use device: {}'.format(self.device)


        