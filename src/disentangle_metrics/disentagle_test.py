import pdb
from src.disentangle_metrics.FVM import FactorVAEMetric
from src.disentangle_metrics.SAP import SAP
from src.disentangle_metrics.DCI import metric_dci
from src.disentangle_metrics.MIG import MIGMetric

def estimate_all_distenglement(data_loader,
                               model,
                               disent_batch_size,
                               disent_num_train,
                               disent_num_test,
                               loss_fn,
                               continuous_factors):
    results = {}
    disent_result = FactorVAEMetric(data_loader,
                                    model=model,
                                    batch_size=100,
                                    num_train=800,
                                    loss_fn=loss_fn)
    results['factor'] = disent_result

    disent_result = SAP(data_loader, model,
                        batch_size=100,
                        num_train=100,
                        num_test=50,
                        loss_fn=loss_fn,
                        continuous_factors=continuous_factors)
    results['sap'] = disent_result

    disent_result = metric_dci(dataset=data_loader,
                               model=model,
                               num_train=100,
                               num_test=50,
                               batch_size=100,
                               loss_fn=loss_fn,
                               continuous_factors=continuous_factors)
    results['dci'] = {}
    results['dci']['train_err'] = disent_result[0]
    results['dci']['test_err'] = disent_result[1]
    results['dci']['disent'] = disent_result[2]
    results['dci']['comple'] = disent_result[3]

    disent_result = MIGMetric(dataset=data_loader,
                              model=model,
                              batch_size=100,
                              num_train=100,
                              loss_fn=loss_fn)
    results['mig'] = disent_result
    return results

