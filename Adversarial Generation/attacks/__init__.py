from future.standard_library import install_aliases
install_aliases()
from urllib import parse as urlparse

import pickle
import numpy as np
import os
import time

from .cleverhans_wrapper import generate_fgsm_examples, generate_jsma_examples, generate_bim_examples
from .carlini_wrapper import generate_carlini_l2_examples, generate_carlini_li_examples, generate_carlini_l0_examples,\
    generate_EAD_examples,generate_EADEN_examples,generate_carlini_l2_adaptive_examples,generate_carlini_l2_adaptive_l2_examples,\
        generate_carlini_l2_adaptive_no_l2_examples,generate_EAD_adaptive_examples,generate_EADEN_adaptive_examples#,generate_carlini_l2_l1_examples,
from .deepfool_wrapper import generate_deepfool_examples, generate_universal_perturbation_examples
#from .adaptive.adaptive_adversary import generate_adaptive_carlini_l2_examples
from .pgd.pgd_wrapper import generate_pgdli_examples


# TODO: replace pickle with .h5 for Python 2/3 compatibility issue.
def maybe_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, use_cache=False, verbose=True, attack_log_fpath=None):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        print ("Loading adversarial examples from [%s]." % os.path.basename(x_adv_fpath))
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath)
        """    
        for chk in range(0,X.shape[0],chunck):
            X_adv = generate_adv_examples(sess, model, x[chk:chk+chunck], y[chk:chk+chunck], X[chk:chk+chunck], Y[chk:chk+chunck], attack_name, attack_params, verbose, attack_log_fpath)
            print ('20 batch comp')#'X_adv.shape)
            if use_cache:
                pickle.dump((X_adv, 0), open(x_adv_fpath, 'wb'))
            X_test_adv_temp_list.extend(X_adv)
        """
        duration = time.time() - time_start
        #X_adv = np.array(X_test_adv_temp_list)
        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

        if use_cache:
            pickle.dump((X_adv, aux_info), open(x_adv_fpath, 'wb'))
    return X_adv, duration


def generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath):
    print (attack_name)
    if attack_name == 'none':
        return X
    elif attack_name == 'fgsm':
        generate_adv_examples_func = generate_fgsm_examples
    elif attack_name == 'jsma':
        generate_adv_examples_func = generate_jsma_examples
    elif attack_name == 'bim':
        generate_adv_examples_func = generate_bim_examples
    elif attack_name == 'carlinil2':
        generate_adv_examples_func = generate_carlini_l2_examples
    elif attack_name == 'carlinil2-adaptive':
        generate_adv_examples_func = generate_carlini_l2_adaptive_examples
    elif attack_name == 'carlinil2-adaptive-l2':
        generate_adv_examples_func = generate_carlini_l2_adaptive_l2_examples
    elif attack_name == 'carlinil2-adaptive-no-l2':
        generate_adv_examples_func = generate_carlini_l2_adaptive_no_l2_examples
    elif attack_name == 'carlinili':
        generate_adv_examples_func = generate_carlini_li_examples
    elif attack_name == 'carlinil0':
        generate_adv_examples_func = generate_carlini_l0_examples
    elif attack_name == 'deepfool':
        generate_adv_examples_func = generate_deepfool_examples
    elif attack_name == 'unipert':
        generate_adv_examples_func = generate_universal_perturbation_examples
    
    elif attack_name == 'pgdli':
        generate_adv_examples_func = generate_pgdli_examples
    elif attack_name == 'eadl1':
        generate_adv_examples_func = generate_EAD_examples
    elif attack_name == 'eaden':
        generate_adv_examples_func = generate_EADEN_examples
    elif attack_name == 'eadl1-adaptive':
        generate_adv_examples_func = generate_EAD_adaptive_examples
    elif attack_name == 'eaden-adaptive':
        generate_adv_examples_func = generate_EADEN_adaptive_examples
    else:
        raise NotImplementedError("Unsuported attack [%s]." % attack_name)
    """
    #elif attack_name == 'adaptive_carlini_l2':
    #    generate_adv_examples_func = generate_adaptive_carlini_l2_examples
    elif attack_name == 'carlinil2_l1':
        generate_adv_examples_func = generate_carlini_l2_l1_examples
    elif attack_name == 'carlinil2_l1_avg':
        generate_adv_examples_func = generate_carlini_l2_l1_avg_examples
    """
    X_adv = generate_adv_examples_func(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath)

    return X_adv

