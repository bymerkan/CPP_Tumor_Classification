# OS
import os

# Torch and TorchVision
import torch
import torchvision
import torchvision.models as m
import torch.nn.functional as F

# ArgParse
import argparse

"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = torch.nn.Linear(100,256)
        self.layer_2 = torch.nn.Linear(256,1)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return x

traced_net = torch.jit.trace(Net(), torch.randn(1,100))

torch.jit.save(traced_net, "/Users/computer/Documents/torch-cpp/models/net.pt")

"""

all_models = m.list_models()

# Create Parser
parser = argparse.ArgumentParser(description='Image Segmentation Project')

parser.add_argument('-a','-arch', dest="architecture", metavar='architecture', default='resnet50',
                    help='model architecture: ' +
                        ' | '.join(all_models) +
                        ' (default: resnet50)')

# main() function
def main():    

    # dataset = torchvision.datasets.MNIST(root="../data/", download = True)

    all_models = m.list_models()
    
    args = parser.parse_args()


    if(args.architecture in all_models):

        model_path = "/Users/computer/Documents/torch-cpp/models/" + args.architecture + ".pt"

        if(not os.path.exists(model_path)):
        
            all_models = m.list_models(module=m.segmentation)

            model = m.get_model(args.architecture, weights="DEFAULT")

            # Edit model before save

            traced_net = torch.jit.trace(model, torch.randn(5,3,256,256), strict=False)

            torch.jit.save(traced_net, str(model_path))

            print("[INFO] --- Model saved succesfully.")

        else:

            """
            model = torch.load(model_path)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    print (name, param.data.size())
            """
            
            print("[INFO] --- Model exists.")

    else:
        print("[ERROR] --- Model did not found in torchvision library. Available models are listed below.\n")
        print(all_models)
        raise(ValueError)
    

if __name__ == '__main__':
    main()