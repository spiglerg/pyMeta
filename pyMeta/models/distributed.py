"""
Variants of the algorithms in `models', that use mpi4py to synchronize gradients across workers that work on different
tasks in each meta-batch.

IMPORTANT: this has only been tested lightly. It seems to be working, but it is still performing sub-optimally.
"""

import numpy as np

from pyMeta.core.meta_learner import GradBasedMetaLearner
from pyMeta.models.fomaml import FOMAMLMetaLearner
from pyMeta.models.reptile import ReptileMetaLearner
from pyMeta.models.seq_fomaml import SeqFOMAMLMetaLearner

from mpi4py import MPI
comm = MPI.COMM_WORLD


class DistributedGradBasedMetaLearner(GradBasedMetaLearner):
    def initialize(self, session):
        super().initialize(session)
        self.current_initial_parameters = comm.bcast(self.current_initial_parameters, root=0)

    def update(self, list_of_final_parameters, **kwargs):
        # Average gradients / parameters for the current worker, in case each worker runs more than a task in
        # the meta-batch
        avg_final = []
        for variables in zip(*list_of_final_parameters):
            avg_final.append(np.mean(variables, axis=0))

        avg_final = comm.gather(avg_final, root=0)
        if comm.rank == 0:
            super().update(avg_final, **kwargs)
        self.current_initial_parameters = comm.bcast(self.current_initial_parameters, root=0)


class DistributedFOMAMLMetaLearner(DistributedGradBasedMetaLearner, FOMAMLMetaLearner):
    pass


class DistributedReptileMetaLearner(DistributedGradBasedMetaLearner, ReptileMetaLearner):
    pass


class DistributedSeqFOMAMLMetaLearner(DistributedGradBasedMetaLearner, SeqFOMAMLMetaLearner):
    pass
