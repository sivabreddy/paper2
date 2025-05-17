from Main import Pre_processing,read
import HFGSO_DRN.run
import ResNet.run
import DCNN.DCNN_run
import Focal_Net.Focal_net
import Panoptic_model.Panoptic
from prop_DMO import DeepMaxout
def callmain(dts,tr_p):
    acc,sen,spe=[],[],[]
    Pre_processing.process()
    Feat = read.read_data()
    Label = read.read_label()
    ################### Calling Methods #################
    print("\n Proposed HGSO-based DRN..")
    DeepMaxout.Dmax(Feat,Label,tr_p,acc,sen,spe)
    HFGSO_DRN.run.classify(Feat, Label, tr_p, acc, sen, spe)
    ResNet.run.classify(Feat,Label,tr_p,acc,sen,spe)
    Focal_Net.Focal_net.callmain(Feat,Label,tr_p,acc,sen,spe)
    Panoptic_model.Panoptic.classify(Feat,Label,tr_p,acc,sen,spe)
    print("\n Please Wait..")
    DCNN.DCNN_run.callmain(Feat,Label,tr_p,acc,sen,spe)
    print(acc,sen,spe)
    print("\n Done.")
    return acc,sen,spe

# dts = "Prostate MRI"
# tr_p = 0.7
#
# callmain(dts,tr_p)