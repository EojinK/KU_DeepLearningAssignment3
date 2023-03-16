# KU_DeepLearningAssignment3
This is an assignment3 I did while taking a deep learning course at Korea University. 

In this assignment, I trained “UNet” model with “Pascal VOC 2012” datasets, and conducted image segmentation.
• Optimize parameters with Adam optimizer and cross Entropy Loss


## 1. Description of my code
### A. Unet_skeleton.py: 
```python
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        ########## fill in the blanks (Hint : check out the channel size in lecture)
        self.convDown1 = conv(in_channels, 64)
        self.convDown2 = conv(64,128)
        self.convDown3 = conv(128,256)
        self.convDown4 = conv(256,512)
        self.convDown5 = conv(512,1024)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convUp4 = conv(1024,512)
        self.convUp3 = conv(512,256)
        self.convUp2 = conv(256,128)
        self.convUp1 = conv(128,64)
        self.convUp_fin = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.convDown1(x)
        x = self.maxpool(conv1)
        conv2 = self.convDown2(x)
        x = self.maxpool(conv2)
        conv3 = self.convDown3(x)
        x = self.maxpool(conv3)
        conv4 = self.convDown4(x)
        x = self.maxpool(conv4)
        conv5 = self.convDown5(x)
        x = self.upsample(conv5)
        x = torch.cat([x, conv4], dim=1) #######fill in here ####### hint : concatenation (Lecture slides)
        x = self.convUp4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1) #######fill in here ####### hint : concatenation (Lecture slides)
        x = self.convUp3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1) #######fill in here ####### hint : concatenation (Lecture slides)
        x = self.convUp2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1) #######fill in here ####### hint : concatenation (Lecture slides)
        x = self.convUp1(x)
        out = self.convUp_fin(x)

        return out
```
In class Unet's ‘def __init__’, I stacked convDown layers from channel size 64 to 1024 according to the Unet picture in the project slide. After maxpool and upsample I stacked convUp layers from 1024 to 64. In ‘def forward’, I concatenated conv4 to conv1 respectively using torch.cat.
resnet_encoder_unet_skeleton.py: Since nothing changed in class ResidualBlock and ‘def __init__’ of class UNetWithResnet50Encoder, I used the code I wrote in project2. I wrote code using torch.cat in ‘def forward’ of class UNetWithResnet50Encoder. (Similar to Unet_skeleton.py)
### B. modules_skeleton.py: 
```python
def train_model(trainloader, model, criterion, optimizer,scheduler, device):
    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        from datetime import datetime

        inputs = inputs.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        criterion = criterion.cuda()
        ##########################################
        ############# fill in here -> train
        ####### Hint :
        ####### 1. Get the output out of model, and Get the Loss
        ####### 3. optimizer
        ####### 4. backpropagation
        
        outputs = model(inputs)
        # print(masks.shape, outputs.shape)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
        #########################################
```

```python
def get_loss_train(model, trainloader, criterion, device):
    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, (inputs, labels) in enumerate(trainloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device = device, dtype = torch.int64)
            inputs = inputs.float()
            ##########################################
            ############# fill in here -> (same as validation, just printing loss)
            ####### Hint :
            ####### Get the output out of model, and Get the Loss
            ####### Think what's different from the above
            
            outputs = model(inputs)
            # print(masks.shape, outputs.shape)
            loss = criterion(outputs, labels)
            # optimizer.zero_grad()
            # loss.backward()
            # Update weights
            # optimizer.step()
            #########################################
```
```python
 def val_model(model, valloader, criterion, device, dir):           
    for batch, (inputs, labels) in enumerate(valloader):
        with torch.no_grad():

            inputs = inputs.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            ##########################################
            ############# fill in here -> (validation)
            ####### Hint :
            ####### Get the output out of model, and Get the Loss
            ####### Think what's different from the above
            
            # print(image_v.shape, mask_v.shape) 
            val_output = model(inputs)
            total_val_loss = total_val_loss + criterion(val_output, labels).cpu().item()
            # print('out', val_output.shape)
            val_output = torch.argmax(val_output, dim=1).float()
            # stacked_img = torch.cat((stacked_img, val_output))
            #########################################
            
                for j in range(temp.shape[0]):
                    for k in range(temp.shape[1]):
                        ##########################################
                        ############# fill in here 
                        ####### Hint :
                        ####### convert segmentation mask into r,g,b (both for image and predicted result)
                        ####### image should become temp_rgb, result should become temp_label
                        ####### You should use cls_invert[]
                        
                        temp_rgb = [cls_invert[j], cls_invert[k], 3]
                        temp_label = [j, k, 3]
                        #########################################
```
In ‘def train_model’, I set output to model(inputs) and loss to criterion(outputs, labels). And I backpropagated through backward(). Similarly in ‘def get_loss_train’ I got outputs and loss. In val_model I converted temp and temp_l to rgb respectively and stored them in temp_rgb and temp_label.
main_skeleton.py: I wrote code to initialize the model with UNet or resnet_encoder_unet. I set the loss function to CrossEntropyLoss and used adam as optimizer. Finally, I loaded the given model parameters.

### C. resnet_encoder_unet_skeleton.py:
```python
###########################################################################
# Code overlaps with previous assignments : Implement the "bottle neck building block" part.
# Hint : Think about difference between downsample True and False. How we make the difference by code?
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.layer = nn.Sequential(
                ##########################################
                ############## fill in here
                # Hint : use these functions (conv1x1, conv3x3)
                conv1x1(in_channels, middle_channels, 2, 0),
                conv3x3(middle_channels, middle_channels, 1, 1),
                conv1x1(middle_channels, out_channels, 1, 0)
                #########################################
            )
            self.downsize = conv1x1(in_channels, out_channels, 2, 0)

        else:
            self.layer = nn.Sequential(
                ##########################################
                ############# fill in here
                conv1x1(in_channels, middle_channels, 1, 0),
                conv3x3(middle_channels, middle_channels, 1, 1),
                conv1x1(middle_channels, out_channels, 1, 0)
                #########################################
            )
            self.make_equal_channel = conv1x1(in_channels, out_channels, 1, 0)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        if self.downsample:
            out = self.layer(x)
            x = self.downsize(x)
            return self.activation(out + x)
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.make_equal_channel(x)
            return self.activation(out + x)
```

### D. main_skeleton.py
```python
##### fill in here #####
##### Hint : Initialize the model (Options : UNet, resnet_encoder_unet)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_set = 1

# Loss Function
##### fill in here -> hint : set the loss function #####
criterion = nn.CrossEntropyLoss()

# Optimizer
##### fill in here -> hint : set the Optimizer #####
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

# parameters
epochs = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

##### fill in here #####
##### Hint : load the model parameter, which is given
model.load_state_dict(torch.load(PATH, map_location=device))
```
```python
    if epoch % 4 == 0:
        savepath2 = savepath1 + str(epoch) + ".pth"
        ##### fill in here #####
        ##### Hint : save the model parameter
        torch.save(model.state_dict(), savepath2)

```
## 2. Results
![image](https://user-images.githubusercontent.com/60259747/225663834-f062838d-ad4b-4990-bbbe-2ee04d33af34.png)
I've been working hard on the code, trying to solve the problem, but not getting any results. The following is a screenshot of the execution result of main_skeleton.py. After this screenshot the Anaconda prompt stopped working.

## 3. Discussion
I think the Anaconda prompt stopped working because there was an error in my code that I didn't recognize, rather than my lack of computing power. I'm sad that I couldn't complete the assignment, and I want to try again if I get a chance.
