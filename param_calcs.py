# parameter calculators

def voc_gcn(N, L):
  """
  (L+2)*d**2 for MP layers, and 2 layer MLP at end
  """
  N*=.98
  return round((N/(L+2))**0.5)

def voc_drew(N, L):
  """
  (L**2+L+4)/2 * d**2 for MP layers, and 2 layer MLP at end
  """
  N*=.975
  return round((N/(0.5*(L**2+L+4)))**0.5)

def pept_gcn(N, L):
  """
  L*d**2 for MP layers, and 2 layer MLP at end
  """
  N*=.98
  return round((N/L)**0.5)

def pept_drew(N, L):
  """
  (L**2+L)/2 * d**2 for MP layers
  """
  N*=.975
  return round((N/(0.5*(L**2+L)))**0.5)

def pcqm_gcn(N, L):
  """
  (L+1+0.5)*d**2 for MP layers, a 1-layer MLP at end, and ~0.5d^2 for node embedding
  """
  N*=.98
  return round((N/(L+1+0.5))**0.5)

def pcqm_drew(N, L):
  """
  (L**2+L+2+1)/2 * d**2 for MP layers, 1-layer MLP at end, and ~0.5d^2 for node embedding
  """
  N*=.975
  return round((N/(0.5*(L**2+L+2+1)))**0.5)

calc_dict = {
  'voc_gcn': voc_gcn,
  'voc_drew': voc_drew,
  'pept_gcn': pept_gcn,
  'pept_drew': pept_drew,
  'pcqm_gcn': pcqm_gcn,
  'pcqm_drew': pcqm_drew,
}
