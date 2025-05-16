import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
import torchvision


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.amp import autocast

from model import AutoEncoder
import utils
from distributions import Normal
import datasets
from train import test, init_processes, test_vae_fid

import networkx as nx
from itertools import combinations

def get_dag(
            logging,
            model,
            args,
            eval_args,
        ) -> None:
    """
    Extracts the DAG from the model's causal layer and plots it.
    
    Args:
        logging: Logger object for logging information.
        model: The model containing the causal layer.
        args: Argument parser containing model and dataset configurations.

    Returns:
        None
    """
    logging.info('evaluating DAG')
    assert 'concept' in args.arch_flag, 'DAG is only supported for concept models.'
    model.eval()
    c = model.module.causal_layer
    obs_adj = (
        c.pooler(c.obs_weight.detach().unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    )
    ivn_adj = (
        c.pooler(c.ivn_weight.detach().unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    )
    adj = torch.minimum(obs_adj, ivn_adj)

    # use single threshold to exract maximal DAG
    weighted_dag = adj
    weighted_dag.fill_diagonal_(0)
    weighted_dag = weighted_dag.cpu().numpy()
    threshs = weighted_dag[weighted_dag > 0].flatten()
    threshs.sort()
    for thresh in threshs:
        thresh_dag = nx.DiGraph(weighted_dag >= thresh)
        if nx.is_directed_acyclic_graph(thresh_dag):
            break

    # first get digraph, then maximal acyclic subgraph
    digraph = (adj > adj.T).to(bool)
    weighted_digraph = torch.zeros_like(adj)
    weighted_digraph[digraph] = adj[digraph]
    weighted_digraph = weighted_digraph.cpu().numpy()

    threshs = weighted_digraph[weighted_digraph > 0].flatten()
    threshs.sort()
    for thresh in threshs:
        acyclic_digraph = nx.DiGraph(weighted_digraph >= thresh)
        if nx.is_directed_acyclic_graph(acyclic_digraph):
            break                       

    # Plot the three DiGraphs
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    labels = [c for c in args.concepts if c != "obs"]
    mapping = {i: label for i, label in enumerate(labels)}

    # Plot each graph in its respective subplot
    thresh_dag = nx.relabel_nodes(thresh_dag, mapping)
    pos1 = nx.spring_layout(thresh_dag)
    nx.draw(
        thresh_dag,
        pos1,
        ax=ax1,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        arrows=True,
    )
    ax1.set_title("single-threshold DAG")

    digraph = nx.DiGraph(digraph.cpu().numpy())
    digraph = nx.relabel_nodes(digraph, mapping)
    pos2 = nx.spring_layout(digraph)
    nx.draw(
        digraph,
        pos2,
        ax=ax2,
        with_labels=True,
        node_color="lightgreen",
        node_size=500,
        arrows=True,
    )
    ax2.set_title("DiGraph")

    acyclic_digraph = nx.relabel_nodes(acyclic_digraph, mapping)
    pos3 = nx.spring_layout(acyclic_digraph)
    nx.draw(
        acyclic_digraph,
        pos3,
        ax=ax3,
        with_labels=True,
        node_color="lightpink",
        node_size=500,
        arrows=True,
    )
    ax3.set_title("maximal acyclic DiGraph")

    plt.tight_layout()
    dir_path = f"{eval_args.save}/dags"
    logging.info('Saving DAGs at %s', dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"{dir_path}/dags.png", dpi=150, bbox_inches="tight")
    plt.close()

def sample_constant_noise(
        logging,
        model,
        args,
        eval_args,
        image_name: str,
        temp: float = 1.0,
    ) -> None:
    """
    Samples a noise vector and then use it to generate an image from each concept.

    Args:
        logging: Logger object for logging information.
        model: The model containing the causal layer.
        args: Argument parser containing model and dataset configurations.
        eval_args: Argument parser containing evaluation configurations.
        image_name: Name of the image file to save.
        temp: Temperature for sampling.
    
    Returns:
        None
    """

    logging.info('sampling with constant noise')
    assert 'concept' in args.arch_flag, 'Constant noise is only supported for concept models.'

    concepts = args.concepts
    num_concepts = len(concepts) - 1 # Exclude 'obs' concept
    assert num_concepts > 0, 'Constant noise is only supported for models with more than one concept.'

    num_samples = 1
    z0_size = [num_samples, model.module.args.eps_dim * model.module.args.eps_in_width] + model.module.z0_size[1:]
    dist = Normal(mu=torch.zeros(z0_size).cuda(), log_sigma=torch.zeros(z0_size).cuda(), temp=temp)
    z, _ = dist.sample()

    fig, axes = plt.subplots(num_concepts, num_concepts, figsize=(num_concepts * 3, num_concepts * 3), squeeze=False)
    cmap = 'gray'

    # Combines every pair of concepts. Does not include obs.
    for i, concept_1 in enumerate(concepts[1:]):
        for j, concept_2 in enumerate(concepts[1:]):

            batch_label = [concept_1, concept_2] if concept_1 != concept_2 else [concept_1,]

            logits = model.module.sample(
                num_samples=num_samples,
                t=temp,
                batch_label=batch_label,
                z=z,
            )

            output = model.module.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                        else output.sample()
            output_img = output_img[0]
            axes[i, j].imshow(output_img.permute(1, 2, 0).detach().cpu().numpy(), cmap=cmap)
            axes[i, j].axis('off')

    for i, concept in enumerate(concepts[1:]):
        axes[-1, i].axis('on')
        axes[-1, i].set_xticks([])
        axes[-1, i].set_yticks([])
        axes[-1, i].set_xlabel(concept, fontsize=24)
    
        axes[i, 0].axis('on')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(concept, fontsize=24)
    
    plt.tight_layout()
    file_path = os.path.join(eval_args.save, image_name)
    logging.info('Saving constant noise samples at %s', file_path)
    if not os.path.exists(eval_args.save):
        os.makedirs(eval_args.save)
    plt.savefig(file_path, dpi=150)
    plt.close()
