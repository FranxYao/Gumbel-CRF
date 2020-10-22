import numpy as np

class TrainingLog(object):
  def __init__(self, config):
    self.model_name = config.model_name
    self.output_path = config.output_path

    self.log = {}
    for n in config.log_info[config.model_name]:
      self.log[n] = []
    return 


  def update(self, output_dict):
    """Update the log"""
    for l in self.log: 
      if(l in output_dict): self.log[l].append(output_dict[l])
    return

  def print(self):
    """Print out the log"""
    s = ""
    # for l in self.log: s += "%s: mean = %.4g, var = %.4g " %\
    #   (l, np.average(self.log[l]), np.var(self.log[l]))
    for l in self.log: s += "%s %.4g\t" % (l, np.average(self.log[l]))
    print(s)
    print("")
    return 

  def write(self, ei, log_metrics=None):
    """Write the log for current epoch"""
    log_path = self.output_path + "epoch_%d.log" % ei
    print("Writing epoch log to %s ... " % log_path)
    with open(log_path, "w") as fd:
      log_len = len(self.log[list(self.log.keys())[0]])
      for i in range(log_len):
        for m in self.log:
          if(log_metrics == None): 
            fd.write("%s: %.4g\t" % (m, self.log[m][i]))
          else:
            if(m in log_metrics): fd.write("%s: %.4f\t" % (m, self.log[m][i]))
        fd.write("\n")
    return 

  def reset(self):
    """Reset the log"""
    for l in self.log: 
        self.log[l] = []
    return