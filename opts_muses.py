import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------ #
    # Overall settings                                                     #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='train | test | test_frame | test_online | eval')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--segment_size',
        type=int,
        default=150)
    parser.add_argument(
        '--anchors',
        type=str,
        default='4,9,18,37,75,112,150')
    parser.add_argument(
        '--seed',
        default=52,
        type=int,
        help='Random seed for reproducibility.')

    # ------------------------------------------------------------------ #
    # Dataset settings                                                     #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--num_of_class',
        type=int,
        default=26,
        help='Number of action classes INCLUDING the background class. '
             'MUSES has 25 action classes → set to 26.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='npy',
        help='Feature file format: pickle | h5 | npz | npz_i3d | pt | npy')
    parser.add_argument(
        '--data_rescale',
        default=False,
        action='store_true')
    parser.add_argument(
        '--predefined_fps',
        default=None,
        type=float)
    parser.add_argument(
        '--rgb_only',
        default=False,
        action='store_true',
        help='Use only the RGB stream (feat_dim halved internally).')

    # Annotation file — single MUSES JSON, no split placeholder needed.
    parser.add_argument(
        '--video_anno',
        type=str,
        default='./data/muses_v1.0.json')

    # Feature directories for npy format (one .npy file per video).
    # Pass the directory path; dataset.py appends "<video_id>.npy" automatically.
    parser.add_argument(
        '--video_feature_all_train',
        type=str,
        default='./data/',
        help='Directory containing per-video .npy feature files for training.')
    parser.add_argument(
        '--video_feature_all_test',
        type=str,
        default='./data/',
        help='Directory containing per-video .npy feature files for validation/test.')

    # Run / experiment identifiers
    parser.add_argument(
        '--setup',
        type=str,
        default='',
        help='Experiment setup tag appended to output file names.')
    parser.add_argument(
        '--exp',
        type=str,
        default='muses01',
        help='Experiment name used in checkpoint and result file names.')
    parser.add_argument(
        '--split',
        type=str,
        default='1',
        help='Dataset split index (used only for multi-split datasets like EGTEA).')

    # ------------------------------------------------------------------ #
    # Network architecture                                                 #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--feat_dim',
        type=int,
        default=1024,
        help='Total input feature dimension. The model splits this in half for its '
             'two-branch reduction (n_feature//2 per branch). '
             'MUSES .npy features are 1024-dim → each branch sees 512. '
             'Use 2048 or 4096 for two-stream RGB+Flow features.')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1024,
        help='Transformer embedding dimension d_model.')
    parser.add_argument(
        '--out_dim',
        type=int,
        default=26,
        help='Output class dimension — must equal num_of_class (25 actions + 1 background).')
    parser.add_argument(
        '--enc_layer',
        type=int,
        default=3)
    parser.add_argument(
        '--enc_head',
        type=int,
        default=8)
    parser.add_argument(
        '--dec_layer',
        type=int,
        default=5)
    parser.add_argument(
        '--dec_head',
        type=int,
        default=4)

    # -- DSE (Dual-Scale Temporal Encoder) --
    parser.add_argument(
        '--DSE',
        default=False,
        action='store_true',
        help='Enable Dual-Scale Temporal Encoder. '
             'When omitted, MYNET runs as the baseline without DSE.')

    # -- DIoU regression loss --
    parser.add_argument(
        '--diou',
        default=False,
        action='store_true',
        help='Add DIoU loss on the regressed (start, end) intervals.')
    parser.add_argument(
        '--diou_weight',
        type=float,
        default=1.0,
        help='Coefficient for the DIoU loss term.')

    # -- GLH (Gaussian Latent History) --
    parser.add_argument(
        '--GLH',
        default=False,
        action='store_true',
        help='Enable Gaussian Latent History cross-attention after the encoder.')
    parser.add_argument(
        '--glh_gaussians',
        type=int,
        default=8,
        help='Number of Gaussian primitives K in the GLH bank.')
    parser.add_argument(
        '--glh_kl_weight',
        type=float,
        default=1e-4,
        help='Weight for the GLH KL regularisation term.')

    # ------------------------------------------------------------------ #
    # Training settings                                                    #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Mini-batch size. A100 80 GB can comfortably handle 512–1024 with AMP.')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--epoch',
        type=int,
        default=5)
    parser.add_argument(
        '--lr_step',
        type=int,
        default=3)

    # ------------------------------------------------------------------ #
    # Post-processing / evaluation                                         #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--alpha',
        type=float,
        default=1,
        help='Weight on the classification loss.')
    parser.add_argument(
        '--beta',
        type=float,
        default=1,
        help='Weight on the regression loss.')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5)
    parser.add_argument(
        '--pptype',
        type=str,
        default='net',
        help='Post-processing type: nms | net (SuppressNet).')
    parser.add_argument(
        '--pos_threshold',
        type=float,
        default=0.5)
    parser.add_argument(
        '--sup_threshold',
        type=float,
        default=0.1)
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1)
    parser.add_argument(
        '--inference_subset',
        type=str,
        default='val',
        help='Subset to evaluate on. MUSES uses "val"; THUMOS uses "test".')
    parser.add_argument(
        '--soft_nms',
        type=float,
        default=0.3)

    # ------------------------------------------------------------------ #
    # Output file paths                                                    #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--video_len_file',
        type=str,
        default='./output/video_len_{}.json')
    parser.add_argument(
        '--proposal_label_file',
        type=str,
        default='./output/proposal_label_{}.h5')
    parser.add_argument(
        '--suppress_label_file',
        type=str,
        default='./output/suppress_label_{}.h5')
    parser.add_argument(
        '--suppress_result_file',
        type=str,
        default='./output/suppress_result.h5')
    parser.add_argument(
        '--frame_result_file',
        type=str,
        default='./output/frame_result.h5')
    parser.add_argument(
        '--result_file',
        type=str,
        default='./output/result_proposal.json')
    parser.add_argument(
        '--wterm',
        type=bool,
        default=False,
        help='Spin-wait after training (useful for keeping GPU warm on some clusters).')

    args = parser.parse_args()
    return args
