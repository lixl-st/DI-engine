import enum
import torch
import torch.nn.functional as F
from ding.torch_utils import flatten, sequence_mask
from ding.utils.data import default_collate
from dizoo.distar.envs import MAX_SELECTED_UNITS_NUM

MASK_INF = -1e9
EPS = 1e-9


def padding_entity_info(traj_data, max_entity_num):
    # traj_data.pop('map_name', None)
    entity_padding_num = max_entity_num - len(traj_data['entity_info']['x'])
    if 'entity_embeddings' in traj_data.keys():
        traj_data['entity_embeddings'] = torch.nn.functional.pad(
            traj_data['entity_embeddings'], (0, 0, 0, entity_padding_num), 'constant', 0
        )

    for k in traj_data['entity_info'].keys():
        traj_data['entity_info'][k] = torch.nn.functional.pad(
            traj_data['entity_info'][k], (0, entity_padding_num), 'constant', 0
        )
    if 'action_info' in traj_data:
        su_padding_num = MAX_SELECTED_UNITS_NUM - traj_data['teacher_logit']['selected_units'].shape[0]

        traj_data['mask']['selected_units_mask'] = sequence_mask(
            traj_data['selected_units_num'].unsqueeze(dim=0), max_len=MAX_SELECTED_UNITS_NUM
        ).squeeze(dim=0)
        traj_data['action_info']['selected_units'] = torch.nn.functional.pad(
            traj_data['action_info']['selected_units'],
            (0, MAX_SELECTED_UNITS_NUM - traj_data['action_info']['selected_units'].shape[-1]), 'constant', 0
        )

        traj_data['behaviour_logp']['selected_units'] = torch.nn.functional.pad(
            traj_data['behaviour_logp']['selected_units'], (
                0,
                su_padding_num,
            ), 'constant', MASK_INF
        )

        traj_data['teacher_logit']['selected_units'] = torch.nn.functional.pad(
            traj_data['teacher_logit']['selected_units'], (
                0,
                entity_padding_num,
                0,
                su_padding_num,
            ), 'constant', MASK_INF
        )
        traj_data['teacher_logit']['target_unit'] = torch.nn.functional.pad(
            traj_data['teacher_logit']['target_unit'], (0, entity_padding_num), 'constant', MASK_INF
        )

        traj_data['mask']['selected_units_logits_mask'] = sequence_mask(
            traj_data['entity_num'].unsqueeze(dim=0) + 1, max_len=max_entity_num + 1
        ).squeeze(dim=0)
        traj_data['mask']['target_units_logits_mask'] = sequence_mask(
            traj_data['entity_num'].unsqueeze(dim=0), max_len=max_entity_num
        ).squeeze(dim=0)

    return traj_data


def collate_fn_learn(traj_batch):
    # data list of list, with shape batch_size, unroll_len
    # find max_entity_num in data_batch
    max_entity_num = max(
        [len(traj_data['entity_info']['x']) for traj_data_list in traj_batch for traj_data in traj_data_list]
    )

    # padding entity_info in observation, target_unit, selected_units, mask
    traj_batch = [
        [padding_entity_info(traj_data, max_entity_num) for traj_data in traj_data_list]
        for traj_data_list in traj_batch
    ]

    data = [default_collate(traj_data_list, allow_key_mismatch=True) for traj_data_list in traj_batch]

    batch_size = len(data)
    unroll_len = len(data[0]['step'])
    data = default_collate(data, dim=1)

    new_data = {}
    for k, val in data.items():
        if k in ['spatial_info', 'entity_info', 'scalar_info', 'entity_num', 'entity_location', 'hidden_state',
                 'value_feature']:
            new_data[k] = flatten(val)
        else:
            new_data[k] = val
    new_data['aux_type'] = batch_size
    new_data['batch_size'] = batch_size
    new_data['unroll_len'] = unroll_len
    return new_data


def entropy_error(target_policy_probs_dict, target_policy_log_probs_dict, mask, head_weights_dict):
    total_entropy_loss = 0.
    entropy_dict = {}
    for head_type in ['action_type', 'queued', 'delay', 'selected_units', 'target_unit', 'target_location']:
        ent = -target_policy_probs_dict[head_type] * target_policy_log_probs_dict[head_type]
        if head_type == 'selected_units':
            ent = ent.sum(dim=-1) / (
                EPS + torch.log(mask['selected_units_logits_mask'].float().sum(dim=-1) + 1).unsqueeze(-1)
            )  # normalize
            ent = (ent * mask['selected_units_mask']).sum(-1)
            ent = ent.div(mask['selected_units_mask'].sum(-1) + EPS)
        elif head_type == 'target_unit':
            # normalize by unit
            ent = ent.sum(dim=-1) / (EPS + torch.log(mask['target_units_logits_mask'].float().sum(dim=-1) + 1))
        else:
            ent = ent.sum(dim=-1) / torch.log(torch.FloatTensor([ent.shape[-1]]).to(ent.device))
        if head_type not in ['action_type', 'delay']:
            ent = ent * mask['actions_mask'][head_type]
        entropy = ent.mean()
        entropy_dict['entropy/' + head_type] = entropy.item()
        total_entropy_loss += (-entropy * head_weights_dict[head_type])
    return total_entropy_loss, entropy_dict


def kl_error(
    target_policy_log_probs_dict, teacher_policy_logits_dict, mask, game_steps, action_type_kl_steps, head_weights_dict
):
    total_kl_loss = 0.
    kl_loss_dict = {}

    for head_type in ['action_type', 'queued', 'delay', 'selected_units', 'target_unit', 'target_location']:
        target_policy_log_probs = target_policy_log_probs_dict[head_type]
        teacher_policy_logits = teacher_policy_logits_dict[head_type]

        teacher_policy_log_probs = F.log_softmax(teacher_policy_logits, dim=-1)
        teacher_policy_probs = torch.exp(teacher_policy_log_probs)
        kl = teacher_policy_probs * (teacher_policy_log_probs - target_policy_log_probs)

        kl = kl.sum(dim=-1)
        if head_type == 'selected_units':
            kl = (kl * mask['selected_units_mask']).sum(-1)
        if head_type not in ['action_type', 'delay']:
            kl = kl * mask['actions_mask'][head_type]
        if head_type == 'action_type':
            flag = game_steps < action_type_kl_steps
            action_type_kl = kl * flag
            action_type_kl_loss = action_type_kl.mean()
            kl_loss_dict['kl/extra_at'] = action_type_kl_loss.item()
        kl_loss = kl.mean()
        total_kl_loss += (kl_loss * head_weights_dict[head_type])
        kl_loss_dict['kl/' + head_type] = kl_loss.item()
    return total_kl_loss, action_type_kl_loss, kl_loss_dict


class ScoreCategories(enum.IntEnum):
  """Indices for the `score_by_category` observation's second dimension."""
  none = 0
  army = 1
  economy = 2
  technology = 3
  upgrade = 4


def compute_battle_score(obs):
    if obs is None:
        return 0.
    score_details = obs.observation.score.score_details
    killed_mineral, killed_vespene = 0., 0.
    for s in ScoreCategories:
        killed_mineral += getattr(score_details.killed_minerals, s.name)
        killed_vespene += getattr(score_details.killed_vespene, s.name)
    battle_score = killed_mineral + 1.5 * killed_vespene
    return battle_score

def l2_distance(a, b, min=0, max=0.8, threshold=5, spatial_x=160):
    x0 = a % spatial_x
    y0 = a // spatial_x
    x1 = b % spatial_x
    y1 = b // spatial_x
    l2 = torch.sqrt((torch.square(x1 - x0) + torch.square(y1 - y0)).float())
    cost = (l2 / threshold).clamp_(min=min, max=max)
    return cost

def levenshtein_distance(behaviour, target, behaviour_extra=None, target_extra=None, extra_fn=None):
    r"""
    Overview:
        Levenshtein Distance(Edit Distance)

    Arguments:
        Note:
            N1 >= 0, N2 >= 0

        - behaviour (:obj:`torch.LongTensor`): shape[N1]
        - target (:obj:`torch.LongTensor`): shape[N2]
        - behaviour_extra (:obj:`torch.Tensor or None`)
        - target_extra (:obj:`torch.Tensor or None`)
        - extra_fn (:obj:`function or None`): if specified, the distance metric of the extra input data

    Returns:
        - (:obj:`torch.FloatTensor`) distance(scalar), shape[1]

    Test:
        torch_utils/network/tests/test_metric.py
    """
    assert (isinstance(behaviour, torch.Tensor) and isinstance(target, torch.Tensor))
    assert behaviour.dtype == target.dtype, f'bahaviour_dtype: {behaviour.dtype}, target_dtype: {target.dtype}'
    assert (behaviour.device == target.device)
    assert (type(behaviour_extra) == type(target_extra))
    if not extra_fn:
        assert (not behaviour_extra)
    N1, N2 = behaviour.shape[0], target.shape[0]
    assert (N1 >= 0 and N2 >= 0)
    if N1 == 0 or N2 == 0:
        distance = max(N1, N2)
    else:
        dp_array = torch.zeros(N1 + 1, N2 + 1).float()
        dp_array[0, :] = torch.arange(0, N2 + 1)
        dp_array[:, 0] = torch.arange(0, N1 + 1)
        for i in range(1, N1 + 1):
            for j in range(1, N2 + 1):
                if behaviour[i - 1] == target[j - 1]:
                    if extra_fn:
                        dp_array[i, j] = dp_array[i - 1, j - 1] + extra_fn(behaviour_extra[i - 1], target_extra[j - 1])
                    else:
                        dp_array[i, j] = dp_array[i - 1, j - 1]
                else:
                    dp_array[i, j] = min(dp_array[i - 1, j] + 1, dp_array[i, j - 1] + 1, dp_array[i - 1, j - 1] + 1)
        distance = dp_array[N1, N2]
    return torch.as_tensor(distance).to(behaviour.device)


def hamming_distance(behaviour, target):
    r'''
    Overview:
        Hamming Distance

    Arguments:
        Note:
            behaviour, target are also boolean vector(0 or 1)

        - behaviour (:obj:`torch.LongTensor`): behaviour input, shape[B, N], while B is the batch size
        - target (:obj:`torch.LongTensor`): target input, shape[B, N], while B is the batch size

    Returns:
        - distance(:obj:`torch.LongTensor`): distance(scalar), the shape[1]

    Shapes:
        - behaviour & target (:obj:`torch.LongTensor`): shape :math:`(B, N)`, \
            while B is the batch size and N is the dimension

    Test:
        torch_utils/network/tests/test_metric.py
    '''
    assert (isinstance(behaviour, torch.Tensor) and isinstance(target, torch.Tensor))
    assert behaviour.dtype == target.dtype, f'bahaviour_dtype: {behaviour.dtype}, target_dtype: {target.dtype}'
    assert (behaviour.device == target.device)
    assert (behaviour.shape == target.shape)
    return behaviour.ne(target).sum(dim=-1).float()
