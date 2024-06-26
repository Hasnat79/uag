import json

with open ("/scratch/user/hasnat.md.abdullah/uag/data/paper_version/ssbd_test_list_fixed.json",'r') as f:
  ssbd = json.load(f)

output=[]
for sample in ssbd:
  id = sample[0]
  info = sample[2]
  data = [id,info]
  output.append(data)



with open("ssbd_paper_version.json", 'w') as f:
  json.dump(output,f,indent=4)

with open("/scratch/user/hasnat.md.abdullah/uag/data/paper_version/uag_oops_dataset_v1_start_time_fixed.json",'r') as f: 
  oops = json.load(f)

out ={}
for id, info in oops.items():
  data = {}
  data['start_time'] = info['start_time']
  data['end_time'] = info['end_time']
  data['description'] = info['description']
  out[id] = data

with open("oops_uag_paper_version.json", 'w') as f:
  json.dump(out,f,indent=4)
