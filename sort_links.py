fl = open("facebook-links.txt.anon", "r")

links = []
with open("facebook-links-sorted.txt.anon", "w") as f:
  for line in fl.readlines():
    els = line.strip().split()
    if els[2] == "\\N":
      links.append((line, float('inf')))
    else:
      links.append((line, int(els[2])))
    
  for el in sorted(links, key=lambda x: x[1]):
    f.write(el[0])