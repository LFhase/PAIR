# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from tkinter.filedialog import test
import numpy as np
from numpy.linalg import norm

top_percentile = 0.9
adjust_coe = 1

def get_kl_div(losses, preference):
    pair_score = losses.dot(preference)
    return pair_score

def get_losses(record):
    if "groupdro" in record['args']['output_dir'] and 'penalty' in record.keys():
        record['loss'] = record['loss']
        record.pop("penalty")
    if 'nll' in record.keys():
        erm_loss = record['nll']
    if 'mu_rl' in record.keys():
        pass
    if "vrex_penalty" in record.keys() and "IRM_penalty" in record.keys():
        losses = np.array([erm_loss,record["IRM_penalty"],record["vrex_penalty"]])
    elif "nvrex_penalty" in record.keys() and "nIRM_penalty" in record.keys():
        losses = np.array([erm_loss,record["nIRM_penalty"],record["nvrex_penalty"]])
    elif 'penalty' in record.keys():
        ood_loss = record['penalty']
        losses = np.array([erm_loss,ood_loss])
    else:
        if 'disc_loss' in record.keys() and 'gen_loss' not in record.keys():
            losses = np.array([(2-record['disc_loss']) if record['disc_loss']<=2 else 3])
            losses = np.array([1e9])
        elif 'gen_loss' in record.keys():
            losses = np.array([1e9])
            if np.abs(record['gen_loss'])>=50:
                losses = np.array([1e9])
            else:
                losses = np.array([(1e9+record['gen_loss']) if record['gen_loss']>=-1e9 else 1e9])
        else:
            losses = np.array([record['loss']])

    return losses

def get_pair_score(record, get_loss=False,preference_base=1e-6):
    if "groupdro" in record['args']['output_dir']and 'penalty' in record.keys():
        record['loss'] = record['loss']
        record.pop("penalty")
    if 'nll' in record.keys():
        erm_loss = record['nll']
    if 'mu_rl' in record.keys():
        pass
    if "vrex_penalty" in record.keys() and "IRM_penalty" in record.keys():
        losses = np.array([erm_loss,record["IRM_penalty"],record["vrex_penalty"]])
        if record["IRM_penalty"] < 0:
            losses[1] *=-adjust_coe
        preference = np.array([preference_base,1e-2,1])
    elif "nvrex_penalty" in record.keys() and "nIRM_penalty" in record.keys():
        losses = np.array([erm_loss,record["nIRM_penalty"],record["nvrex_penalty"]])
        if record["nIRM_penalty"] < 0:
            losses[1] *=-adjust_coe
        preference = np.array([preference_base,1e-2,1])
    elif 'penalty' in record.keys():
        ood_loss = record['penalty']
        losses = np.array([erm_loss,ood_loss])
        if record["penalty"] < 0:
            losses[1] *=-adjust_coe
        preference = np.array([preference_base,1])
    else:
        if 'disc_loss' in record.keys() and 'gen_loss' not in record.keys():
            losses = np.array([(2-record['disc_loss']) if record['disc_loss']<=2 else 3])
        elif 'gen_loss' in record.keys():
            if np.abs(record['gen_loss'])>=50:
                losses = np.array([1e9])
            else:
                losses = np.array([(1e9+record['gen_loss']) if record['gen_loss']>=-1e9 else 1e9])
        else:
            losses = np.array([record['loss']])
        preference = np.array([1]) if len(losses)==1 else np.array([preference_base,1])

    pair_score = get_kl_div(losses,preference)
    if get_loss:
        return -pair_score, losses
    return -pair_score

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        # it implicitly assumes there is a test env, and n-1 training env
        # hence given n envs, it does the eval with all 2-test-env combinations
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None

from domainbed.lib.query import Q
class PAIRIIDAccuracySelectionMethod(SelectionMethod):
    """Model selection according to PAIR score from
        Pareto Invariant Risk Minimization."""
    name = "pair training-domain validation set"
    preference_base=1e-6
    
    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        
        pair_score,losses = get_pair_score(record=record,get_loss=True,preference_base=self.preference_base)
        return {
            'losses': losses,
            'pair_score': pair_score,
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """

        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        num_records = len(test_records)
        
        test_records = test_records.map(self._step_acc)
        # filter out worst top_percentile% records in val acc to avoid trivial case
        # return test_records.argmax('val_acc')
        train_accs = [r['val_acc'] for r in test_records]
        train_acc_bar = (np.max(train_accs)-np.min(train_accs))*0.8+np.min(train_accs)
        pair_scores = [r['pair_score'] for r in test_records]
        pair_score_bar = (np.max(pair_scores)-np.min(pair_scores))*0.9+np.min(pair_scores)

        if "coloredmnist" in run_records[0]['args']['dataset'].lower()or ("irm" in run_records[0]['args']['output_dir']):
            test_records = Q(test_records[-5:])
        else:
            test_records = Q(test_records[-10:])

        return test_records.argmax(lambda x: x['val_acc']*(-1 if x['pair_score']<pair_score_bar else 1))

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        tmp_records = []
        for r in records.group('args.hparams_seed'):
            r = get_test_records(r[1])
            if len(r)>0:
                tmp_records.append(r)
        self.preference_base = 10**int(np.log10(np.mean([np.min([np.abs(get_losses(r)[-1]) for r in rr])  for rr in tmp_records]))-2)
        records = (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
        )

        num_records = len(records)
        # filter out worst top_percentile% records in val acc to avoid trivial case
        train_accs = [r[0]['val_acc'] for r in records]
        train_acc_bar = (np.max(train_accs)-np.min(train_accs))*0.5+np.min(train_accs)
        pair_scores = [r[0]['pair_score'] for r in records]
        pair_score_bar = (np.max(pair_scores)-np.min(pair_scores))*0.9+np.min(pair_scores)
        if "dann" not in records[0][1][0]['args']['output_dir'] and "groupdro" not in records[0][1][0]['args']['output_dir']:
            return records.sorted(key=lambda x: x[0]['pair_score']*(1e8 if x[0]['val_acc']<train_acc_bar else 1))[::-1]
        else:
            return records.sorted(key=lambda x: x[0]['val_acc'])[::-1]
        
        return records.sorted(key=lambda x: x[0]['val_acc']*(-1 if x[0]['pair_score']<pair_score_bar else 1))[::-1]
         

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class PAIROracleSelectionMethod(SelectionMethod):
    """Model selection according to PAIR score from
        Pareto Invariant Risk Minimization."""
    name = "pair test-domain validation set (oracle)"
    preference_base = 1e-6
    

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        if len(record['args']['test_envs']) > 1:
            return None
        test_env = record['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        train_accs = []
        for i in range(1,10):
            if i == test_env or 'env{}_out_acc'.format(i) not in record.keys():
                continue
            train_accs.append(record['env{}_out_acc'.format(i)])
        pair_score,losses = get_pair_score(record=record,get_loss=True,preference_base=self.preference_base)
        return {
            'losses': losses,
            'train_acc': np.mean(train_accs),
            'pair_score': pair_score,
            'val_acc':  record[test_out_acc_key],
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        num_records = len(test_records)
        test_records = test_records.map(self._step_acc)
        train_acc_bar = 0
        train_accs = [r['train_acc'] for r in test_records]
        train_acc_bar = (np.max(train_accs)-np.min(train_accs))*0.1+np.min(train_accs)

        erm_bar = 1
        erm_losses = [r['losses'][0] for r in test_records]
        erm_bar = (np.max(erm_losses)-np.min(erm_losses))*0.8+np.min(erm_losses)

        pair_scores = [r['pair_score'] for r in test_records]
        pair_score_bar = (np.max(pair_scores)-np.min(pair_scores))*0.9+np.min(pair_scores)

        for r in test_records:
            r['train_bar']=train_acc_bar
            r['erm_bar']=erm_bar
            r['pair_score_bar']=pair_score_bar

        if "dann" in run_records[0]['args']['output_dir']: 
            test_records = Q(test_records[-5:]) 
        else:
            test_records = Q(test_records[-10:]) 

        return test_records.argmax('pair_score')
        return test_records[-1]

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        tmp_records = []
        for r in records.group('args.hparams_seed'):
            r = get_test_records(r[1])
            if len(r)>0:
                tmp_records.append(r)

        self.preference_base = 10**int(np.log10(np.mean([np.min([np.abs(get_losses(r)[-1]) for r in rr])  for rr in tmp_records]))-2)
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class PAIRLeaveOneOutSelectionMethod(SelectionMethod):
    """Model selection according to PAIR score from
        Pareto Invariant Risk Minimization."""
    name = "pair leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        pair_scores = np.zeros(n_envs) - 1
        # it implicitly assumes there is a test env, and n-1 training env
        # hence given n envs, it does the eval with all 2-test-env combinations
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
            pair_scores[val_env] = get_pair_score(r)
        
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        pair_score = np.sum(pair_scores) / (n_envs-1)
        return {
            'pair_score': pair_score,
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    
    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()

        if len(step_accs):
            num_records = len(step_accs)
            # filter out worst top_percentile% records in val acc to avoid trivial case
            step_accs = Q(step_accs.sorted(key=lambda x: x['val_acc'])[int(num_records*top_percentile):])
            return step_accs.argmax('pair_score')
        else:
            return None
