class Config:
    gpu = '4'
    train_dir = './OASIS_L2R/All/'
    val_dir = './OASIS_L2R/Test/'
    lr = 1e-3
    max_epochs = 500
    sim_loss = 'ncc'
    edge_sim_loss = 'mse'
    alpha = 1
    beta = 0.3
    batch_size = 1
    n_save_epoch = 10
    model_dir = './Checkpoint/'
    save_model_num = 3
    input_channel = 14

    # test model path
    checkpoint_path = "./Checkpoint/..."
