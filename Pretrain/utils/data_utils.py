# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


def get_loader(args):
    splits0 = "/dataset00_BTCV.json"
    splits1 = "/dataset01_BrainTumour.json"
    splits2 = "/dataset02_Heart.json"
    splits3 = "/dataset03_Liver.json"
    splits4 = "/dataset04_Hippocampus.json"
    splits5 = "/dataset05_Prostate.json"
    splits6 = "/dataset06_Lung.json"
    splits7 = "/dataset07_Pancreas.json"
    splits8 = "/dataset08_HepaticVessel.json"
    splits9 = "/dataset09_Spleen.json"
    splits10 = "/dataset10_Colon.json"
    splits11 = "/dataset11_TCIAcovid19.json"
    splits12 = "/dataset12_WORD.json"
    splits13 = "/dataset13_AbdomenCT-1K.json"
    
    list_dir = "./jsons"
    jsonlist0 = list_dir + splits0
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    jsonlist7 = list_dir + splits7
    jsonlist8 = list_dir + splits8
    jsonlist9 = list_dir + splits9
    jsonlist10 = list_dir + splits10
    jsonlist11 = list_dir + splits11
    jsonlist12 = list_dir + splits12
    jsonlist13 = list_dir + splits13
    
    datadir0 = "./dataset/dataset00_BTCV"
    datadir1 = "./dataset/dataset01_BrainTumour"
    datadir2 = "./dataset/dataset02_Heart"
    datadir3 = "./dataset/dataset03_Liver"
    datadir4 = "./dataset/dataset04_Hippocampus"
    datadir5 = "./dataset/dataset05_Prostate"
    datadir6 = "./dataset/dataset06_Lung"
    datadir7 = "./dataset/dataset07_Pancreas"
    datadir8 = "./dataset/dataset08_HepaticVessel"
    datadir9 = "./dataset/dataset09_Spleen"
    datadir10 = "./dataset/dataset10_Colon"
    datadir11 = "./dataset/dataset11_TCIAcovid19"
    datadir12 = "./dataset/dataset12_WORD"
    datadir13 = "./dataset/dataset13_AbdomenCT-1K"
    
    num_workers = 8
    datalist0 = load_decathlon_datalist(jsonlist0, False, "training", base_dir=datadir0)
    print("Dataset 0 BTCV: number of data: {}".format(len(datalist0)))
    new_datalist0 = []
    for item in datalist0:
        item_dict = {"image": item["image"]}
        new_datalist0.append(item_dict)
    
    # datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    # print("Dataset 1 BrainTumour: number of data: {}".format(len(datalist1)))
    # new_datalist1 = []
    # for item in datalist1:
    #     item_dict = {"image": item["image"]}
    #     new_datalist1.append(item_dict)
        
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Heart: number of data: {}".format(len(datalist2)))
    new_datalist2 = []
    for item in datalist2:
        item_dict = {"image": item["image"]}
        new_datalist2.append(item_dict)
        
    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print("Dataset 3 Liver: number of data: {}".format(len(datalist3)))
    new_datalist3 = []
    for item in datalist3:
        item_dict = {"image": item["image"]}
        new_datalist3.append(item_dict)

    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    print("Dataset 4 Hippocampus: number of data: {}".format(len(datalist4)))
    new_datalist4 = []
    for item in datalist4:
        item_dict = {"image": item["image"]}
        new_datalist4.append(item_dict)
    
    # datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    # print("Dataset 5 Prostate: number of data: {}".format(len(datalist5)))
    # new_datalist5 = []
    # for item in datalist5:
    #     item_dict = {"image": item["image"]}
    #     new_datalist5.append(item_dict)
    
    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    print("Dataset 6 Lung: number of data: {}".format(len(datalist6)))
    new_datalist6 = []
    for item in datalist6:
        item_dict = {"image": item["image"]}
        new_datalist6.append(item_dict)

    datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    print("Dataset 7 Pancreas: number of data: {}".format(len(datalist7)))
    new_datalist7 = []
    for item in datalist7:
        item_dict = {"image": item["image"]}
        new_datalist7.append(item_dict)

    datalist8 = load_decathlon_datalist(jsonlist8, False, "training", base_dir=datadir8)
    print("Dataset 8 HepaticVessel: number of data: {}".format(len(datalist8)))
    new_datalist8 = []
    for item in datalist8:
        item_dict = {"image": item["image"]}
        new_datalist8.append(item_dict)  

    datalist9 = load_decathlon_datalist(jsonlist9, False, "training", base_dir=datadir9)
    print("Dataset 9 Spleen: number of data: {}".format(len(datalist9)))
    new_datalist9 = []
    for item in datalist9:
        item_dict = {"image": item["image"]}
        new_datalist9.append(item_dict)

    datalist10 = load_decathlon_datalist(jsonlist10, False, "training", base_dir=datadir10)
    print("Dataset 10 Colon: number of data: {}".format(len(datalist10)))
    new_datalist10 = []
    for item in datalist10:
        item_dict = {"image": item["image"]}
        new_datalist10.append(item_dict)

    datalist11 = load_decathlon_datalist(jsonlist11, False, "training", base_dir=datadir11)
    print("Dataset 11 TCIAcovid19: number of data: {}".format(len(datalist11)))
    new_datalist11 = []
    for item in datalist11:
        item_dict = {"image": item["image"]}
        new_datalist11.append(item_dict)

    datalist12 = load_decathlon_datalist(jsonlist12, False, "training", base_dir=datadir12)
    print("Dataset 12 WORD: number of data: {}".format(len(datalist12)))
    new_datalist12 = []
    for item in datalist12:
        item_dict = {"image": item["image"]}
        new_datalist12.append(item_dict)

    datalist13 = load_decathlon_datalist(jsonlist13, False, "training", base_dir=datadir13)
    print("Dataset 13 AbdomenCT-1K: number of data: {}".format(len(datalist13)))
    new_datalist13 = []
    for item in datalist13:
        item_dict = {"image": item["image"]}
        new_datalist13.append(item_dict)
    
    vallist0 = load_decathlon_datalist(jsonlist0, False, "validation", base_dir=datadir0)
    # vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
    vallist4 = load_decathlon_datalist(jsonlist4, False, "validation", base_dir=datadir4)
    # vallist5 = load_decathlon_datalist(jsonlist5, False, "validation", base_dir=datadir5)
    vallist6 = load_decathlon_datalist(jsonlist6, False, "validation", base_dir=datadir6)
    vallist7 = load_decathlon_datalist(jsonlist7, False, "validation", base_dir=datadir7)
    vallist8 = load_decathlon_datalist(jsonlist8, False, "validation", base_dir=datadir8)
    vallist9 = load_decathlon_datalist(jsonlist9, False, "validation", base_dir=datadir9)
    vallist10 = load_decathlon_datalist(jsonlist10, False, "validation", base_dir=datadir10)
    #vallist11 = load_decathlon_datalist(jsonlist11, False, "validation", base_dir=datadir11)
    vallist12 = load_decathlon_datalist(jsonlist12, False, "validation", base_dir=datadir12)
    vallist13 = load_decathlon_datalist(jsonlist13, False, "validation", base_dir=datadir13)
    
    datalist =  new_datalist0 + new_datalist2 + new_datalist3 + new_datalist4 + new_datalist6 + new_datalist7 + new_datalist8 + new_datalist9 + new_datalist10+ new_datalist11 +new_datalist12 + new_datalist13
    val_files = vallist0 + vallist2 + vallist3 + vallist4 + vallist6 + vallist7 + vallist8 + vallist9 + vallist10 + vallist12 + vallist13
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            # CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            # CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

#     if args.cache_dataset:
#         print("Using MONAI Cache Dataset")
#         train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
#     elif args.smartcache_dataset:
#         print("Using MONAI SmartCache Dataset")
#         train_ds = SmartCacheDataset(
#             data=datalist,
#             transform=train_transforms,
#             replace_rate=1.0,
#             cache_num=2 * args.batch_size * args.sw_batch_size,
#         )
#     else:
#         print("Using generic dataset")
    train_ds = Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader
