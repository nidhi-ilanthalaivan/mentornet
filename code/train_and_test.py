from cifar_train_baseline import train_inception_baseline
from cifar_eval import eval_inception

if __name__ == '__main__':
    print("======================================== Training Started ========================================")
    train_inception_baseline()
    print("========================================  Training Ended  ========================================")
    print("========================================  Testing Started  ========================================")
    eval_inception()
    print("========================================  Testing Ended  ========================================")