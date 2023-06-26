import sys
import os
import numpy as np
import copy
root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(root_path)

from sklearn.preprocessing import normalize
from params import *
from cluster import test_cluster_accuracy, latent_vec_2_cluster_proba
from utils import file_path, eval_data_distribution, latent_vec_2_cluster_proba
import scipy.stats

from progress.bar import Bar


def search_plan(init_proba: np.ndarray, goal_proba: np.ndarray):

    search_levels_dict = {}
    search_action_dict = {}
    search_KL_divergance = {}
    action_prob_dict = {}
    np.set_printoptions(precision=3, suppress=True)
    # print('init_proba: ', init_proba)
    # print('goal_proba: ', goal_proba)


    for i in range(search_level):
        search_levels_dict['{}_level'.format(i)] = []
        search_action_dict['{}_level'.format(i)] = []
        search_KL_divergance['{}_level'.format(i)] = []

    search_levels_dict['0_level'].append(init_proba)
    search_action_dict['0_level'].append([])
    goal_proba += 1e-10
    init_proba += 1e-10
    # search_KL_divergance['0_level'].append(scipy.stats.entropy(init_proba, goal_proba))
    search_KL_divergance['0_level'].append(scipy.stats.entropy(goal_proba, init_proba))

    for i in range(1, search_level):

        level = '{}_level'.format(i)
        pre_level = '{}_level'.format(i - 1)
        probas = search_levels_dict[pre_level]
        for proba_index, proba in enumerate(probas):

            # print('proba: ', proba)

            predicted_probas = predict_proba(init_proba=proba)
            # print('predicted_probas: ', predicted_probas)

            # push_predict_kl = scipy.stats.entropy(predicted_probas['Push'], proba)
            # mate_predict_kl = scipy.stats.entropy(predicted_probas['Mate'], proba)
            push_predict_kl = scipy.stats.entropy(proba + 1e-10, predicted_probas['Push'] + 1e-10)
            mate_predict_kl = scipy.stats.entropy(proba + 1e-10, predicted_probas['Mate'] + 1e-10)
            # print('push_predict_kl: ', push_predict_kl)
            # print('mate_predict_kl: ', mate_predict_kl)

            if push_predict_kl > lower_KL_divergance_threshold:
                search_levels_dict[level].append(predicted_probas['Push'])
                # search_KL_divergance[level].append(scipy.stats.entropy(predicted_probas['Push'], goal_proba))
                search_KL_divergance[level].append(scipy.stats.entropy(goal_proba + 1e-10, predicted_probas['Push'] + 1e-10))
                search_action_dict[level].append(copy.deepcopy(search_action_dict[pre_level][proba_index]))
                search_action_dict[level][-1].append('Push')

            if mate_predict_kl > lower_KL_divergance_threshold:
                search_levels_dict[level].append(predicted_probas['Mate'])
                # search_KL_divergance[level].append(scipy.stats.entropy(predicted_probas['Mate'], goal_proba))
                search_KL_divergance[level].append(scipy.stats.entropy(goal_proba + 1e-10, predicted_probas['Mate'] + 1e-10))
                search_action_dict[level].append(copy.deepcopy(search_action_dict[pre_level][proba_index]))
                search_action_dict[level][-1].append('Mate')

    # print('search_KL_divergance: ', search_KL_divergance)
    # print('search_action_dict: ', search_action_dict)
    # exit()

    def insert_queue(KL_action_pair: list, KL_action_queue: list):
        cur_KL_divergances = []
        for pair in KL_action_queue:
            cur_KL_divergances.append(pair[0])
        cur_KL_divergances = np.array(cur_KL_divergances)
        insert_index = np.sum(KL_action_pair[0] > cur_KL_divergances)
        KL_action_queue.insert(insert_index, KL_action_pair)

        return

    lower_KL_action_queue = []
    higher_KL_action_queue = []
    best_action_pair = None

    for i in range(0, search_level):

        level = '{}_level'.format(i)

        for KL_index, KL_divergance in enumerate(search_KL_divergance[level]):

            KL_action_pair = [KL_divergance, search_action_dict[level][KL_index]]

            if KL_divergance < lower_KL_divergance_threshold:
                insert_queue(KL_action_pair=KL_action_pair, KL_action_queue=lower_KL_action_queue)

            elif KL_divergance < higher_KL_divergance_threshold:
                insert_queue(KL_action_pair=KL_action_pair, KL_action_queue=higher_KL_action_queue)

        if not len(lower_KL_action_queue) == 0:
            best_action_pair = lower_KL_action_queue[0]
            break

    predicted_probs = []

    if best_action_pair is None and not len(higher_KL_action_queue) == 0:
        best_action_pair = higher_KL_action_queue[0]

    if len(lower_KL_action_queue) == 0 and len(higher_KL_action_queue) == 0:
        return None, [], [], {}

    if len(best_action_pair[1]) == 0:
        return 'End', ['End'], [], {}
    else:
        plan = copy.deepcopy(best_action_pair[1])
        plan_copy = copy.deepcopy(plan)
        while len(plan_copy):
            for key, value in search_action_dict.items():
                if plan_copy in value:
                    predicted_probs.append(search_levels_dict[key][value.index(plan_copy)])
            plan_copy.pop()
        predicted_probs.reverse()
        plan.append('End')

        for index, action in enumerate(plan):
            if action != 'End':
                action_prob_dict[action.lower()] = predicted_probs[index]

        return plan[0], plan, predicted_probs, action_prob_dict


def find_end_state_distribution():
    latent_vectors_test = np.load(file_path(file_name=encoded_latent_vectors_npy, file_path=True, split='train'))
    image_labels_test = np.load(file_path(file_name=image_labels_npy, file_path=True, split='train'))

    if len(np.unique(image_labels_test)) == 4:
        ending = end_label
    elif len(np.unique(image_labels_test)) == 11:
        ending = end_states

    for state in ending:
        vector_index_options = np.where(image_labels_test == state)[0]
        summation = np.zeros((cluster_center_num))
        for index in vector_index_options:
            summation = latent_vec_2_cluster_proba(latent_vectors_test[index, :]) + summation

    end_state_distibution = summation / np.sum(summation)
    np.save(file_path(file_name=end_state_distribution_npy, file_path=False, split=None), end_state_distibution)
    return end_state_distibution


def plan_test(end_state_distribution: np.ndarray=None):
    latent_vectors_test = np.load(file_path(file_name=encoded_latent_vectors_npy, file_path=True, split='test'))
    image_labels_test = np.load(file_path(file_name=image_labels_npy, file_path=True, split='test'))

    latent_vectors_train = np.load(file_path(file_name=encoded_latent_vectors_npy, file_path=True, split='train'))
    image_labels_train = np.load(file_path(file_name=image_labels_npy, file_path=True, split='train'))

    # eval_data_distribution(image_labels_test, latent_vectors_test)

    if len(np.unique(image_labels_test)) == 4:
        ending = end_label
    elif len(np.unique(image_labels_test)) == 11:
        ending = end_states

    if isinstance(end_state_distribution, np.ndarray):
        goal_proba = end_state_distribution

    elif end_state_distribution is None:
        goal_proba = np.load(file_path(file_name=end_state_distribution_npy, file_path=True, split=None))

    # goal_proba =np.array([0, 1, 0, 0], dtype=np.float32)
    # print('goal_proba: ', goal_proba)
    # print('latent_vectors_test: ', latent_vectors_test.shape)
    # print('image_labels_test: ', image_labels_test.shape)
    
    average_vectors = {}
    for i in np.unique(image_labels_test):
        i_indices = np.where(image_labels_test == i)[0]
        for index in i_indices:
        # print('latent_vec_2_cluster_proba(latent_vectors_test[i_indices, :]): ', latent_vec_2_cluster_proba(latent_vectors_test[i_indices, :]).shape)
            proba = latent_vec_2_cluster_proba(latent_vectors_test[index, :]).reshape(-1, 4)
            # print('proba: ', proba)
            # exit()
            if i in list(average_vectors.keys()):
                average_vectors[i] += proba
            else:
                average_vectors[i] = proba
        average_vectors[i] /= len(i_indices)
    np.set_printoptions(precision=3, suppress=True)
    # print('test average_vectors: ', average_vectors)

    average_vectors = {}
    for i in np.unique(image_labels_train):
        i_indices = np.where(image_labels_train == i)[0]
        for index in i_indices:
        # print('latent_vec_2_cluster_proba(latent_vectors_test[i_indices, :]): ', latent_vec_2_cluster_proba(latent_vectors_test[i_indices, :]).shape)
            proba = latent_vec_2_cluster_proba(latent_vectors_train[index, :]).reshape(-1, 4)
            # print('proba: ', proba)
            # exit()
            if i in list(average_vectors.keys()):
                average_vectors[i] += proba
            else:
                average_vectors[i] = proba
        average_vectors[i] /= len(i_indices)
    np.set_printoptions(precision=3, suppress=True)
    # print('train average_vectors: ', average_vectors)

    # exit()

    success_num = 0
    first_success_num = 0
    test_num = 330
    tested_indices = []

    sequences_test_num = {}
    sequences_test_success_num = {}
    sequences_test_first_success_num = {}

    bar = Bar('Processing', max=test_num)
    for _ in range(test_num):
        while True:
            rand_index = np.random.randint(latent_vectors_test.shape[0])

            initial_label = image_labels_test[rand_index]
            cur_label = copy.deepcopy(initial_label)
            # if not cur_label == 3:
            #     continue

            cur_latent_vector = latent_vectors_test[rand_index, :]

            if not rand_index in tested_indices:
                break
        tested_indices.append(rand_index)

        Groundtruth_plan = groundtruth_plan[action_type[initial_label]]
        initial_plan = None
        executed_plan = []
        if not action_type[initial_label] in list(sequences_test_num.keys()):
            sequences_test_num[action_type[initial_label]] = 1
            sequences_test_success_num[action_type[initial_label]] = 0
            sequences_test_first_success_num[action_type[initial_label]] = 0

        sequences_test_num[action_type[initial_label]] += 1

        loop_times = 0
        while True:
            proba = latent_vec_2_cluster_proba(cur_latent_vector)
            # test = search_plan(init_proba=proba, goal_proba=goal_proba)
            # print(len(test))
            predicted_action, plan, _, _ = search_plan(init_proba=proba, goal_proba=goal_proba)
            # print('cur_label: ', list(action_type.values())[int(cur_label)])
            # print('plan: ', plan)
            # print('#'*50)

            if not plan is None and initial_plan is None:
                initial_plan = copy.deepcopy(plan)
            if not predicted_action is None:
                executed_plan.append(predicted_action)

            if predicted_action == 'End' or predicted_action is None:
                break

            cur_label, cur_latent_vector = simulate_action(cur_label=cur_label,
                                                           action=predicted_action,
                                                           latent_vectors=latent_vectors_test,
                                                           image_labels=image_labels_test)
            loop_times += 1
            if loop_times > 5:
                break
        # exit()
        # print('executed_plan: ', executed_plan)
        # print('initial_plan: ', initial_plan)
        # print('Groundtruth_plan: ', Groundtruth_plan)
        # exit()

        if initial_plan == executed_plan and initial_plan in Groundtruth_plan:
            sequences_test_first_success_num[action_type[initial_label]] += 1
            first_success_num += 1

        if cur_label in ending and not predicted_action is None:
            sequences_test_success_num[action_type[initial_label]] += 1
            success_num += 1
        bar.next()
    bar.finish()
    print('success rate: ', success_num / test_num)
    print('first success rate: ', first_success_num / test_num)
    print('#' * 50)
    for key, value in sequences_test_num.items():
        print('value: ', value)
        test_num_temp = value
        test_success_num_temp = sequences_test_success_num[key]
        test_first_success_num_temp = sequences_test_first_success_num[key]

        print('test num: ', value)
        print('success num: ', test_success_num_temp)
        print('first success num: ', test_first_success_num_temp)

        print('{} success rate: '.format(key), test_success_num_temp / test_num_temp)
        print('{} first success rate: '.format(key), test_first_success_num_temp / test_num_temp)

    return plan


def simulate_action(cur_label, action, latent_vectors, image_labels):

    Dynamic = {'Push': Push_dynamic, 'Mate': Mate_dynamic}
    next_labels = np.where(Dynamic[action][int(cur_label), :] == 1)[0]

    vector_index_options = np.where(image_labels == next_labels)[0]
    rand_vector_index = np.random.choice(vector_index_options)
    next_label = image_labels[rand_vector_index]
    next_laten_vector = latent_vectors[rand_vector_index, :]

    return next_label, next_laten_vector


def predict_proba(init_proba: np.ndarray):

    assert init_proba.shape[0] == cluster_center_num

    A = np.zeros((label_num, label_num))
    A_push = np.zeros((label_num, label_num))
    A_mate = np.zeros((label_num, label_num))
    A_self = np.eye(label_num)

    for push in Push:
        A[push[0], push[1]] = 1
        A_push[push[0], push[1]] = 1

    for mate in Mate:
        A[mate[0], mate[1]] = 1
        A_mate[mate[0], mate[1]] = 1

    for self_ in Self:
        A[self_[0], self_[1]] = 1
    for self_ in Push_self:
        A_push[self_[0], self_[1]] = 1
    for self_ in Mate_self:
        A_mate[self_[0], self_[1]] = 1

    A = normalize(A, axis=1, norm='l1')
    A_push = normalize(A_push, axis=1, norm='l1')
    A_mate = normalize(A_mate, axis=1, norm='l1')
    # print('A_push: ', np.where(A_push==1))
    # print('A_mate: ', np.where(A_mate==1))


    init_cluster_prob = init_proba

    init_cluster_prob /= np.sum(init_cluster_prob)

    P = np.zeros((cluster_center_num, cluster_center_num))

    P_purity_num = np.load(file_path(file_name=label_nums_npy, file_path=True, split=None))
    P_purity_num = P_purity_num.astype(int)

    P_purity_col_norm = normalize(P_purity_num, axis=0, norm='l1')
    P_purity_row_norm = normalize(P_purity_num, axis=1, norm='l1')
    np.set_printoptions(precision=3, suppress=True)
    # print('P_purity_col_norm: ', P_purity_col_norm)
    # print('P_purity_row_norm: ', P_purity_row_norm)
    # print('P_purity_num: ', P_purity_num)

    action_prob = {'Push': 0, 'Mate': 0}
    P_push = np.zeros((cluster_center_num, cluster_center_num))
    P_mate = np.zeros((cluster_center_num, cluster_center_num))

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            P_entry = 0

            P_entry_push = 0
            P_entry_mate = 0
            for m in range(P_purity_row_norm.shape[1]):
                for n in range(P_purity_row_norm.shape[1]):
                    P_entry += P_purity_row_norm[i, m] * A[m, n] * P_purity_col_norm[j, n]

                    P_entry_push += P_purity_row_norm[i, m] * A_push[m, n] * P_purity_col_norm[j, n]
                    P_entry_mate += P_purity_row_norm[i, m] * A_mate[m, n] * P_purity_col_norm[j, n]

            P[i, j] = P_entry

            P_push[i, j] = P_entry_push
            P_mate[i, j] = P_entry_mate

    P_succesor = init_cluster_prob @ P

    P_succesor_mate = init_cluster_prob @ P_mate
    P_succesor_push = init_cluster_prob @ P_push

    # print('P: ', P)
    # print('#'*25)
    # print('P_mate: ', P_mate)
    # print('P_push: ', P_push)
    # print('P_self: ', P_self)

    # print('init_cluster_prob: ', init_cluster_prob)
    # print('P_succesor: ', P_succesor)
    # print('P_succesor_mate: ', P_succesor_mate)
    # print('P_succesor_push: ', P_succesor_push)
    # print('P_succesor_self: ', P_succesor_self)

    # print('P_succesor_mate: ', np.sum(P_succesor_mate))
    # print('P_succesor_push: ', np.sum(P_succesor_push))
    # print('P_succesor_self: ', np.sum(P_succesor_self))

    # # print('sum: ', P_succesor_mate+P_succesor_push+P_succesor_self)

    # print('P_succesor: ', np.sum(P_succesor))
    # print('#' * 50)
    # exit()

    return {'Push': P_succesor_push, 'Mate': P_succesor_mate}


if __name__ == '__main__':
    end_state_distribution = find_end_state_distribution()
    plan = plan_test(end_state_distribution=None)
    print('plan: ', plan)