from collections import defaultdict
import numpy as np
import os
import pandas as pd
from pprint import PrettyPrinter
import random
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

# TODO: add base dataset class for the following if possible

# HKTV MALL MIX PRODUCT
class HKTVMallMixed(Dataset):
    def __init__(self, directory='../data/mix_products_v2', df_path='../data/mix_products_v2/products.csv', split=None, img_transform=None, text_transform=None, target_transform=None):
        self.original_df = self.read_metadata(df_path)
        self.directory = directory
        self.img_labels = self.split_metadata(split)
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.class_names = self.original_df['category'].unique()
        self.num_classes = self.original_df['category'].nunique()
        self.class_to_idx = {key: idx for idx, key in enumerate(self.class_names)}
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_name = str(self.img_labels.iloc[index]['image'])
        img_path = os.path.join(self.directory, img_name)
        image = read_image(img_path, ImageReadMode.RGB)
        text = self.img_labels.iloc[index]['summary']
        target = self.class_to_idx[self.img_labels.iloc[index]['category']]
        if self.img_transform:
            image = self.img_transform(image)
        if self.text_transform:
            text = self.text_transform(text)
        if self.target_transform:
            target = self.target_transform(target)
        return image, text, target
    
    def split_metadata(self, split):
        if split is None:
            return self.original_df
        train, val, test = 0.7, 0.15, 0.15
        seed = 1172023
        train_dataset = self.original_df.sample(frac=train, random_state=seed)
        val_dataset = self.original_df.drop(train_dataset.index).sample(frac=val / (1 - train), random_state=seed)
        test_dataset = self.original_df.drop(train_dataset.index).drop(val_dataset.index)
        if split == 'train':
            return train_dataset
        if split == 'val':
            return val_dataset
        if split == 'test':
            return test_dataset

    @staticmethod
    def read_metadata(df_path, min_category_count=10):
        df = pd.read_csv(df_path)
        df['category'] = df['cate']
        df.drop(['cate', 'raw'], axis=1, inplace=True)
        return df


# SHOPEE
class Shopee(Dataset):
    def __init__(self, directory='../data/shopee-product-matching/train_images', df_path='../data/shopee-product-matching/', split=None, img_transform=None, text_transform=None, target_transform=None):
        self.original_df = self.read_metadata(df_path)
        self.img_labels = self.split_metadata(df_path, split)
        self.directory = directory
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.class_names = self.original_df['category'].unique()
        self.num_classes = self.original_df['category'].nunique()
        self.class_to_idx = {key: idx for idx, key in enumerate(self.class_names)}

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_name = str(self.img_labels.iloc[index]['image'])
        img_path = os.path.join(self.directory, img_name)
        image = read_image(img_path, ImageReadMode.RGB)
        text = self.img_labels.iloc[index]['summary']
        target = self.class_to_idx[self.img_labels.iloc[index]['category']]
        if self.img_transform:
            image = self.img_transform(image)
        if self.text_transform:
            text = self.text_transform(text)
        if self.target_transform:
            target = self.target_transform(target)
        return image, text, target

    def split_metadata(self, df_path, split):
        # if split is None:
        #     return self.original_df
        # train, val, test = 0.8, 0.1, 0.1
        # seed = 23012023
        # train_dataset = self.original_df.sample(frac=train, random_state=seed)
        # val_dataset = self.original_df.drop(train_dataset.index).sample(frac=val / (1 - train), random_state=seed)
        # test_dataset = self.original_df.drop(train_dataset.index).drop(val_dataset.index)
        # if split == 'train':
        #     return train_dataset
        # if split == 'val':
        #     return val_dataset
        # if split == 'test':
        #     return test_dataset
        if split is None:
            return self.original_df
        
        df = pd.read_csv(df_path + split + "_split.csv")
        df['summary'] = df['title']
        df['category'] = df['label_group']
        df.drop(['title', 'image_phash', 'label_group'], axis=1, inplace=True)

        return df

    @staticmethod
    def read_metadata(df_path):
        df = pd.read_csv(df_path + "all_split.csv").dropna()
        df['summary'] = df['title']
        df['category'] = df['label_group']
        df.drop(['title', 'image_phash', 'label_group'], axis=1, inplace=True)
        return df


# FASHION PRODUCT
class FashProd(Dataset):
    def __init__(self, directory='../data/fashProd/images640', df_path='../data/fashProd/updated_styles.csv', split=None, img_transform=None, text_transform=None, target_transform=None):
        self.original_df = self.read_metadata(df_path)
        self.img_labels = self.split_metadata(split)
        self.directory = directory
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.class_names = self.original_df['category'].unique()
        self.num_classes = self.original_df['category'].nunique()
        self.class_to_idx = {key: idx for idx, key in enumerate(self.class_names)}
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.directory, str(self.img_labels.iloc[index, 0]) + '.jpg')
        image = read_image(img_path, ImageReadMode.RGB)
        text = self.img_labels.iloc[index]['summary']
        label = self.class_to_idx[self.img_labels.iloc[index]['category']]
        if self.img_transform:
            image = self.img_transform(image)
        if self.text_transform:
            text = self.text_transform(text)        
        if self.target_transform:
            label = self.target_transform(label)
        return image, text, label

    def split_metadata(self, split):
        if split is None:
            return self.original_df
        num_data = len(self.original_df)
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        train_size = int(np.floor(0.008 * num_data))
        val_size = int(np.floor(0.001 * num_data))
        test_size = int(np.floor(0.001 * num_data))
        # val_size = (num_data - train_size) // 2
        # test_size = num_data - train_size - val_size
        train_dataset = self.original_df.iloc[idx[:train_size]]
        val_dataset = self.original_df.iloc[idx[train_size:train_size + val_size]]
        test_dataset = self.original_df.iloc[idx[-test_size:]]
        if split == 'train':
            return train_dataset
        if split == 'val':
            return val_dataset
        if split == 'test':
            return test_dataset

    @staticmethod
    def read_metadata(df_path, min_category_count=10):
        cols = ['id', 'productDisplayName', 'productDescriptors-description-value', 'masterCategory', 'subCategory', 'articleType']
        df = pd.read_csv(df_path).loc[:, cols].dropna()
        df['summary'] = df['productDisplayName']
        # df['summary'] = df['productDescriptors-description-value']
        df['category'] = df['masterCategory'] + '-' + df['subCategory'] + '-' + df['articleType']
        
        while True:
            change_cnt = 0
            
            for category, count in df['category'].value_counts().items():
                idx = df['category'] == category
                if count < min_category_count:
                    cates = category.split('-')
                    for i in range(len(cates)-1, -1, -1):
                        if cates[i] != 'Others':
                            cates[i] = 'Others'
                            break
                        elif i != 0:
                            del cates[i]
                    new_cat = '-'.join(cates)
                    if new_cat != category:
                        change_cnt += 1
                    df.loc[idx, 'category'] = new_cat
            
            if change_cnt == 0:
                break
        
        df.drop(['masterCategory', 'subCategory', 'articleType'], axis=1, inplace=True)
        return df


# SIGIR FARFETCH
class SigirFarfetch(Dataset):
    def __init__(self, directory='../data/sigir_farfetch/utils/data/images', df_path='../data/sigir_farfetch/products.parquet', split=None, img_transform=None, text_transform=None, target_transform=None):
        self.original_df = self.read_metadata(df_path)
        self.img_labels = self.split_metadata(split)
        self.directory = directory
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.class_names = self.original_df['category'].unique()
        self.num_classes = self.original_df['category'].nunique()
        self.class_to_idx = {key: idx for idx, key in enumerate(self.class_names)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_name = str(self.img_labels.iloc[index, 0])
        img_dir = [img_name[i: i+2] for i in range(0, len(img_name), 2)]
        img_path = os.path.join(self.directory, *img_dir, img_name + '.jpg')
        image = read_image(img_path, ImageReadMode.RGB)
        text = self.img_labels.iloc[index]['summary']
        target = self.class_to_idx[self.img_labels.iloc[index]['category']]
        if self.img_transform:
            image = self.img_transform(image)
        if self.text_transform:
            text = self.text_transform(text)
        if self.target_transform:
            target = self.target_transform(target)
        return image, text, target

    def split_metadata(self, split):
        if split is None:
            return self.original_df
        num_data = len(self.original_df)
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        train_size = int(np.floor(0.8 * num_data))
        val_size = (num_data - train_size) // 2
        test_size = num_data - train_size - val_size
        train_dataset = self.original_df.iloc[idx[:train_size]]
        val_dataset = self.original_df.iloc[idx[train_size:train_size + val_size]]
        test_dataset = self.original_df.iloc[idx[-test_size:]]
        if split == 'train':
            return train_dataset
        if split == 'val':
            return val_dataset
        if split == 'test':
            return test_dataset
    
    @staticmethod
    def read_metadata(df_path, min_category_count=40):
        cols = ['product_id', 'product_family', 'product_category', 'product_sub_category', 
                'product_brand', 'product_short_description', 'product_highlights',
                'product_image_path']
        df = pd.read_parquet(df_path).loc[:, cols].dropna()
        df['product_highlights'] = df['product_highlights'].map(lambda x: ' '.join(x.strip('[]').split(', ')))
        # df['summary'] = df['product_brand'] + ' [SEP] ' + df['product_short_description'] + ' [SEP] ' + df['product_highlights']
        df['summary'] = df['product_short_description']
        df['category'] = df['product_family'] + '>' + df['product_category']
        idx = df['product_sub_category'] != 'N/D'
        df.loc[idx, 'category'] = df['category'] + '>' + df['product_sub_category'].loc[idx]
        
        while True:
            change_cnt = 0
            
            for cat, cnt in df['category'].value_counts().items():
                idx = df['category'] == cat
                if cnt < min_category_count:
                    cates = cat.split(">")
                    for i in range(len(cates)-1, -1, -1):
                        if cates[i] != "Others":
                            cates[i] = "Others"
                            break
                        elif i != 0:
                            del cates[i]
                    new_cat = ">".join(cates)
                    if new_cat != cat:
                        change_cnt += 1
                    df.loc[idx, 'category'] = new_cat
            
            if change_cnt == 0:
                break
        return df


class ShopeeCL(Dataset):
    def __init__(self, directory='../data/shopee-product-matching/train_images', df_path='../data/shopee-product-matching/', split=None, img_transform=None, text_transform=None, target_transform=None):
        self.original_df = self.read_metadata(df_path)
        self.img_labels = self.split_metadata(df_path, split)
        self.directory = directory
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.class_names = [0, 1]
        self.num_classes = 2
        self.class_to_idx = {key: idx for idx, key in enumerate(self.class_names)}
        self.lg_to_idx = {key: idx for idx, key in enumerate(self.original_df['category'].unique())}
        self.label_groups, self.lg_pid = self.map_pid_to_lg()
    
    def __len__(self):
        return len(self.img_labels)

    def get_positive_pair(self, index, lg):
        positive_samples = set(self.lg_pid[lg]) - set([index])
        if positive_samples:
            return random.choice(list(positive_samples))
        return None

    def get_negative_pair(self, index, lg):
        negative_lg = random.choice(list(self.label_groups - set([lg])))
        return self.lg_pid[negative_lg][0]

    def __getitem__(self, index):
        found = False
        while not found:
            try:
                # 1st product
                img1_path = os.path.join(self.directory, str(self.img_labels.iloc[index]['image']))
                image1 = read_image(img1_path, ImageReadMode.RGB)
                text1 = self.img_labels.iloc[index]['summary']
                label1 = self.lg_to_idx[self.img_labels.iloc[index]['category']]
                if self.img_transform:
                    image1 = self.img_transform(image1)
                if self.text_transform:
                    text1 = self.text_transform(text1)
                if self.target_transform:
                    label1 = self.target_transform(label1)
                
                # 2nd product
                is_positive = random.randint(0, 1)
                self.img_labels['is_positive'].at[index] = is_positive
                if is_positive:
                    pair_index = self.get_positive_pair(index, label1)
                    if pair_index is None:
                        is_positive = 0
                        self.img_labels['is_positive'].at[index] = 0
                        pair_index = self.get_negative_pair(index, label1)
                else:
                    pair_index = self.get_negative_pair(index, label1)

                img2_path = os.path.join(self.directory, str(self.img_labels.iloc[pair_index]['image']))
                image2 = read_image(img2_path, ImageReadMode.RGB)
                text2 = self.img_labels.iloc[pair_index]['summary']
                label2 = self.lg_to_idx[self.img_labels.iloc[pair_index]['category']]
                if self.img_transform:
                    image2 = self.img_transform(image2)
                if self.text_transform:
                    text2 = self.text_transform(text2)
                if self.target_transform:
                    label2 = self.target_transform(label2)
                
                found = True
                    
            except IndexError as e:
                # pick another index if cannot find index
                # print(e)
                index = random.randint(0, len(self.img_labels) - 1)

        return (image1, image2), (text1, text2), (label1, label2)
    
    def split_metadata(self, df_path, split):
        if split is None:
            self.original_df['is_positive'] = 0
            return self.original_df
        
        df = pd.read_csv(df_path + split + "_split.csv").dropna()
        df['summary'] = df['title']
        df['category'] = df['label_group']
        df.drop(['title', 'image_phash', 'label_group'], axis=1, inplace=True)

        df['is_positive'] = 0
        return df
        
    def map_pid_to_lg(self):
        label_groups = set()
        lg_pid = defaultdict(list)
        
        for i, lg in enumerate(self.img_labels['category'].to_list()):
            lg = self.lg_to_idx[lg]
            lg_pid[lg].append(i)
            label_groups.add(lg)
        return label_groups, dict(lg_pid)

    @staticmethod
    def read_metadata(df_path):
        df = pd.read_csv(df_path + "all_split.csv").dropna()
        df['summary'] = df['title']
        df['category'] = df['label_group']
        df.drop(['title', 'image_phash', 'label_group'], axis=1, inplace=True)
        return df


class ShopeeTL(Dataset):
    def __init__(self, directory='../data/shopee-product-matching/train_images', df_path='../data/shopee-product-matching/', split=None, img_transform=None, text_transform=None, target_transform=None):
        self.original_df = self.read_metadata(df_path)
        self.img_labels = self.split_metadata(df_path, split)
        self.directory = directory
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.lg_to_idx = {key: idx for idx, key in enumerate(self.original_df['category'].unique())}
        self.label_groups, self.lg_pid = self.map_pid_to_lg()
    
    def __len__(self):
        return len(self.img_labels)

    def get_positive_pair(self, index, lg):
        positive_samples = set(self.lg_pid[lg]) - set([index])
        if positive_samples:
            return random.choice(list(positive_samples))
        return None

    def get_negative_pair(self, index, lg):
        negative_lg = random.choice(list(self.label_groups - set([lg])))
        return self.lg_pid[negative_lg][0]

    def __getitem__(self, index):
        found = False
        while not found:
            try:
                # 1st product
                img1_path = os.path.join(self.directory, str(self.img_labels.iloc[index]['image']))
                image1 = read_image(img1_path, ImageReadMode.RGB)
                text1 = self.img_labels.iloc[index]['summary']
                label1 = self.lg_to_idx[self.img_labels.iloc[index]['category']]
                if self.img_transform:
                    image1 = self.img_transform(image1)
                if self.text_transform:
                    text1 = self.text_transform(text1)
                if self.target_transform:
                    label1 = self.target_transform(label1)
                
                # 2nd product    
                p_index = self.get_positive_pair(index, label1)
                if p_index is None:
                    p_index = index
                
                img2_path = os.path.join(self.directory, str(self.img_labels.iloc[p_index]['image']))
                image2 = read_image(img2_path, ImageReadMode.RGB)
                text2 = self.img_labels.iloc[p_index]['summary']
                label2 = self.lg_to_idx[self.img_labels.iloc[p_index]['category']]
                if self.img_transform:
                    image2 = self.img_transform(image2)
                if self.text_transform:
                    text2 = self.text_transform(text2)
                if self.target_transform:
                    label2 = self.target_transform(label2)

                # 3rd product    
                n_index = self.get_negative_pair(index, label1)
                if n_index is None:
                    n_index = index

                img3_path = os.path.join(self.directory, str(self.img_labels.iloc[p_index]['image']))
                image3 = read_image(img3_path, ImageReadMode.RGB)
                text3 = self.img_labels.iloc[p_index]['summary']
                label3 = self.lg_to_idx[self.img_labels.iloc[p_index]['category']]
                if self.img_transform:
                    image3 = self.img_transform(image3)
                if self.text_transform:
                    text3 = self.text_transform(text3)
                if self.target_transform:
                    label3 = self.target_transform(label3)
                
                found = True
                    
            except IndexError as e:
                # pick another index if cannot find index
                print(e)
                index = random.randint(0, len(self.img_labels) - 1)

        return (image1, image2, image3), (text1, text2, text3), (label1, label2, label3)
    
    def split_metadata(self, df_path, split):
        if split is None:
            return self.original_df
        
        df = pd.read_csv(df_path + split + "_split.csv")
        df['summary'] = df['title']
        df['category'] = df['label_group']
        df.drop(['title', 'image_phash', 'label_group'], axis=1, inplace=True)

        return df
    
    def map_pid_to_lg(self):
        label_groups = set()
        lg_pid = defaultdict(list)
        
        for i, lg in enumerate(self.img_labels['category'].to_list()):
            lg = self.lg_to_idx[lg]
            lg_pid[lg].append(i)
            label_groups.add(lg)
        return label_groups, dict(lg_pid)

    @staticmethod
    def read_metadata(df_path):
        df = pd.read_csv(df_path + "all_split.csv").dropna()
        df['summary'] = df['title']
        df['category'] = df['label_group']
        df.drop(['title', 'image_phash', 'label_group'], axis=1, inplace=True)
        return df

if __name__ == '__main__':
    data = ShopeeCL(split="train")
    # print(*[data[i] for i in range(5)])
    for img, text, y in data:
        try:
            assert isinstance(text[0], str)
            assert isinstance(text[1], str)
            assert isinstance(y[0], int)
            assert isinstance(y[1], int)
        except TypeError as e:
            print(e)
    