import os
import re
import shutil
import subprocess
import time

import torch

from detect import detect_dataset
from model import EAST


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
    """evaluate a saved model from .pth

    Args:
        model_name (str): the .pth model file
        test_img_path (str): the image path to test
        submit_path (str): the submit path used by evaluation tool
        save_flag (bool, optional): whether to save the predictions. Defaults to True.
    """
    print("epoch name: ", model_name)
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)
    os.chdir(submit_path)
    res = subprocess.getoutput("zip -q submit.zip *.txt")
    res = subprocess.getoutput("mv submit.zip ../")
    os.chdir("../")
    res = subprocess.getoutput(
        "python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip"
    )
    print(res)
    os.remove("./submit.zip")
    print("eval time is {}".format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit_path)


def eval_torch_model(model, test_img_path, submit_path, save_flag=True):
    """evaluate a torch.model object directely, and return the results

    Args:
        model_name (torch.model): the torch.model object to be evaluated
        test_img_path (str): the image path to test
        submit_path (str): the submit path used by evaluation tool
        save_flag (bool, optional): whether to save the predictions. Defaults to True.
    """
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)
    os.chdir(submit_path)
    res = subprocess.getoutput("zip -q submit.zip *.txt")
    res = subprocess.getoutput("mv submit.zip ../")
    os.chdir("../")
    res = subprocess.getoutput(
        "python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip"
    )
    print(res)
    os.remove("./submit.zip")
    print("eval time is {}".format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit_path)

    # extract python float number from output string
    # e.g. res is  {"precision": 0.7593778591033852, "recall": 0.7992296581608088, "hmean": 0.7787942763312221, "AP": 0}, then
    # return 0.7593778591033852, 0.7992296581608088, 0.7787942763312221
    _res = re.split("\n|, |: ", res)
    acc, recall, f1 = float(_res[1]), float(_res[3]), float(_res[5])
    return acc, recall, f1


if __name__ == "__main__":
    model_name = "./pths/model_epoch_580_02Apr_8288.pth"
    test_img_path = os.path.abspath("../ICDAR_2015/test_img")
    submit_path = "./submit"
    eval_model(model_name, test_img_path, submit_path)
