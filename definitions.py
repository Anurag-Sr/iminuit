import argparse
import logging

def parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dist", help="distribution type")
    parser.add_argument("-m", "--mean", help="mean")
    parser.add_argument("-s", "--std", help="std")
    parser.add_argument("-dc", "--decay", help="eponential decay constant")
    parser.add_argument("-sl", "--slope", help = "slope for linear fit")

    return parser

def logger():
    # Create and configure logger
    logging.basicConfig(filename="newfile.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to INFO
    logger.setLevel(logging.INFO)
    
    return logger