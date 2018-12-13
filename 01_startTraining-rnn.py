import src.rnn_helper as rnn

def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    # training parameters
    epochs = 100
    lr = 1e-4
    
    pStreamlineData = 'data/ismrm_csd_fa015_curated_without_fp.vtk'
    
    model = rnn.train(pStreamlineData=pStreamlineData, noEpochs = epochs, lr = lr)
    
    model.save('final_model.h5')

    
if __name__ == "__main__":
    main()
