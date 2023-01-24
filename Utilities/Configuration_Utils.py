
def GetModifiedConf(keys=[],vals=[]):
    base_conf = {}
    tmp=base_conf.copy()
    for (key,val) in zip(keys,vals):
        tmp[key]=val
    return tmp

class ConfClass():
    def __init__(self):
        # misc + logging
        self.GPU = -1
        self.LogToWandb=True
        self.WadbUsername='EnterYourUserName'
        self.ProjectName='SDA_Experiments_Project'
        self.ExpName='SDA_Experiment'
        self.MonitorTraining=True
        self.ValMonitoringFactor=10

        # dataset
        self.Src='M'
        self.Tgt='U'
        self.TaskName= self.Src+'_to_'+self.Tgt
        self.SamplesPerClass=7
        self.UnlabeledTgtClasses=0
        self.TaskObjective='CE' # CrossEntropy(CE), L1, L2

        # optimizer hyperparameters
        self.BatchSize=256
        self.NumberOfBatches = 10000
        self.LearningRate=1
        self.Optimizer='SGD'
        self.WD=1e-3

        # model hyperparameters
        self.Coeffs=[1,1,1,1,1]
        self.UdaMethod='CORAL'
        self.NumberOfNearestNeighbours = -1
        self.KernelScale = 'Auto'
        self.Method='SDA_IO'



##

def GetParser(parser):
    parser.add_argument("--Src", type=str, default="U",
                        help="Source domain", choices=['U', 'M', 'A', 'W', 'D'])
    parser.add_argument("--Tgt", type=str, default="M",
                        help="Target domain", choices=['U', 'M', 'A', 'W', 'D'])
    parser.add_argument("--SamplesPerClass", type=int, default=3,
                        help="SamplesPerClass", choices=[1, 3, 5, 7])
    parser.add_argument("--Method", type=str, default='SDA_IO',
                        help="Method, our methods is denoted by SDA_IO", choices=['SDA_IO', 'CCSA', 'dSNE', 'NEM'])
    parser.add_argument("--GPU_ID", type=int, default=-1,
                        help="GPU_ID, -1 for CPU")
    parser.add_argument("--LogToWandb", type=bool, default=False,
                        help="Log the results to Weights and Biases")
    parser.add_argument("--return_counts", type=bool, default=True)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    return parser


def GetConfFromArgs(args):
    hp = ConfClass()
    hp.Src=args.Src
    hp.Tgt =args.Tgt
    hp.GPU=args.GPU_ID
    hp.Method=args.Method
    hp.SamplesPerClass= args.SamplesPerClass
    hp.LogToWandb=args.LogToWandb

    FieldsForName = ['Src', 'Tgt', 'SamplesPerClass', 'Method']
    hp.ExpName = '_'.join(['%s="%s"' % (f, hp.__getattribute__(f)) for f in FieldsForName])

    if args.Src in ['M','U']:
        hp.Optimizer = 'SGD'
        hp.LearningRate=1e-4
        hp.BatchSize=128
        hp.NumberOfBatches=50000
        hp.WD=1e-3

        # ------- for testing (fast convergence) -------
        # hp.Optimizer = 'Adadelta'
        # hp.LearningRate = 1
        # hp.NumberOfBatches = 1000

    if args.Src in ['A','W','D']:
        hp.Optimizer = 'SGD'
        hp.LearningRate = 1e-4
        hp.BatchSize = 32
        hp.NumberOfBatches = 50000
        hp.WD = 1e-4

    return hp