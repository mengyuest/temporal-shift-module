import os
train_ls=open("training.csv").readlines()
val_ls=open("validation.csv").readlines()
act_ls=open("actions.csv").readlines()

id2act={}
id2verb={}
id2noun={}
act2id={}
verb2id={}
noun2id={}

for l in act_ls[1:]:
    act_idx=int(l.split(", ")[0])
    verb_idx=int(l.strip().split(",")[-2])
    noun_idx=int(l.strip().split(",")[-1])
    word=l.strip().split(",")[1].strip()
    word_verb=word.split("_")[0]
    word_noun=word.split("_")[1]

    if act_idx not in id2act:
        id2act[act_idx]=word
        act2id[word]=act_idx
    if verb_idx not in id2verb:
        id2verb[verb_idx] = word_verb
        verb2id[word_verb] = verb_idx
    if noun_idx not in id2noun:
        id2noun[noun_idx] = word_noun
        noun2id[word_noun] = noun_idx

# print stats
print("#acti:%d"%len(id2act))
print("#verb:%d"%len(verb2id))
print("#noun:%d"%len(noun2id))

with open("classIndActions.txt", "w") as f:
    for i, action in enumerate(act2id):
        f.write("%d,%s\n"%(i, action))

with open("classIndVerbs.txt", "w") as f:
    for i, verb in enumerate(verb2id):
        f.write("%d,%s\n"%(i, verb))

with open("classIndNouns.txt", "w") as f:
    for i, noun in enumerate(noun2id):
        f.write("%d,%s\n"%(i, noun))

with open("classMappings.txt", "w") as f:
    verbs_iter_d = {}
    nouns_iter_d = {}
    for it, verb in enumerate(verb2id):
        verbs_iter_d[verb] = it
    for it, noun in enumerate(noun2id):
        nouns_iter_d[noun] = it
    print(verbs_iter_d, nouns_iter_d)
    for i, action in enumerate(act2id):
        verb, noun = action.split("_")
        f.write("%d,%d,%d,%s,%s,%s\n" % (i, verbs_iter_d[verb], nouns_iter_d[noun], action, verb, noun))

exit("stop earlier, check code for the purpose")
# stat1=[]
# stat2=[]
# stat3=[]
# for l in train_ls:
#     items = l.strip().split(", ")
#     stat1.append(items[-3])
#     stat2.append(items[-2])
#     stat3.append(items[-1])
# print("stat1:%d"%(len(set(stat1))))
# print("stat2:%d"%(len(set(stat2))))
# print("stat3:%d"%(len(set(stat3))))

# verb, noun, action

ls = {"training":train_ls, "validation": val_ls}
ofs = {"actions":2, "verbs":0, "nouns":1}
for topic in ["actions", "verbs", "nouns"]:
    offset = ofs[topic]
    for split in ["training", "validation"]:
        with open("%s_%s.txt"%(split, topic), "w") as f:
            for l in ls[split]:
                items = l.strip().split(", ")
                f.write("%s %s %s %s\n"%(items[1], int(items[2]), int(items[3]), int(items[4+offset])))
