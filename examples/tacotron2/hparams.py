''' configs file '''
import sys
sys.path.append('..')
from examples.tokenisers.symbols import symbols


class hparams:
    ''' configs '''
    text_cleaners = ['english_cleaners']

    # Preprocessing parameters
    num_mels = 80
    num_freq = 513
    sample_rate = 22050
    frame_length_ms = 50
    frame_shift_ms = 12.5
    preemphasis = 0.85
    min_level_db = -100
    ref_level_db = 20
    power = 1.5
    gl_iters = 100
    fmin = 125
    fmax = 7600
    hop_length = 256
    win_length = 1024

    # Model Parameters

    symbols_embedding_dim = 512

    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    # Decoder parameters
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_steps = 1000
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 256

    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5

    lr = 0.002
    epoch_num = 2000
    batch_size = 2
    test_batch_size = 1
    mask_padding = True
    p = 10  # mel spec loss penalty

    max_text_len = 189
    n_symbols = len(symbols)
