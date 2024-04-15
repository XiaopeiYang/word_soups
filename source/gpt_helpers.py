from source.loading_helpers import load_gpt_descriptions
import torch
import torch.nn.functional as F
from source.utils import get_text_labels, replace_underscores
from config import Fungi_dataset

descriptor_fname_dict = {
    'ImageNet':'descriptors_imagenet',
    Fungi_dataset:'fungi_descriptions/fungi_descriptions11_model7b'#change
}

between_text_dict = {
    'ImageNet':'',
    Fungi_dataset:', a type of fungi'
}

suffix_dict = {
    'ImageNet':'.',
    Fungi_dataset:', a type of fungi'
}

manual_prompts = [    
    'a photo of a {}.',
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.']

def get_openai_manual_prompt_template_centroids(
    classnames, 
    model, 
    dataset,
    tokenizer,
    n_descriptors=-1
):
    suffix = suffix_dict[dataset]
    f_text_list = []
    if n_descriptors > 0:
        _manual_prompts = manual_prompts[:n_descriptors]
    else:
        _manual_prompts = manual_prompts
        
    for prompt in _manual_prompts:
        text_label_list = get_text_labels(classnames, prompt)
        text_label_list_with_suffix = []
        for s in text_label_list:
            assert s[-1] == '.'
            text_label_list_with_suffix.append(s[:-1] + suffix)
        print('Example: {}'.format(text_label_list_with_suffix[0]))
        
        prompt_init = model.shallow_prompt.prompt_init
        
        if prompt[:len(prompt_init)] == prompt_init:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                f_text = encode_text_wrapper(model, text_label_list_with_suffix, tokenizer,use_raw=False)
        else:
            # this is for prompts that do not start with coop initialization
            text = tokenizer(text_label_list_with_suffix)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                f_text = model.encode_text(text.cuda())
            f_text = F.normalize(f_text.float())
            
        f_text_list.append(f_text)
    
    return F.normalize(torch.stack(f_text_list).mean(0))

def get_gpt_descriptions(dataset):
    descriptor_fname = 'gpt_descriptors/' + descriptor_fname_dict[dataset]
    hparams = {}
    hparams['category_name_inclusion'] = 'prepend'
    hparams['descriptor_fname'] = descriptor_fname
    hparams['after_text'] = hparams['label_after_text'] = '.'
    hparams['before_text'] = 'a photo of a '
    hparams['label_before_text'] = ""
    hparams['between_text'] = between_text_dict[dataset] + ', '
    hparams['apply_descriptor_modification'] = True
    gpt_descriptions, _, _ = load_gpt_descriptions(hparams, classes_to_load=None)
    return gpt_descriptions

def get_gpt_raw_descriptions(dataset):    
    # For other datasets, build the file path normally
    descriptor_fname = 'gpt_descriptors/' + descriptor_fname_dict[dataset]
    hparams = {}
    hparams['category_name_inclusion'] = 'prepend'
    hparams['descriptor_fname'] = descriptor_fname
    hparams['after_text'] = hparams['label_after_text'] = '.'
    hparams['before_text'] = 'a photo of a '
    hparams['label_before_text'] = ""
    hparams['between_text'] = between_text_dict[dataset] + ', '
    hparams['apply_descriptor_modification'] = True
    _, _,gpt_raw_descriptions = load_gpt_descriptions(hparams, classes_to_load=None)
    return gpt_raw_descriptions

def transform_classname(c):
    '''
    There is a slight miss match in imagenet classnames.
    Handle that here.
    '''
    classname = c
    
    if c == 'newt':
        classname = 'smooth newt'
    elif c == 'tights':
        classname = 'maillot'
       
    return classname

def encode_text_wrapper(model, string_text, tokenizer,use_raw):
    emb, eofs = model.shallow_prompt.get_text_embeddings_from_tokenized_string(
                model, tokenizer(string_text).cuda(), use_raw
            )
    return F.normalize(
        model.encode_text(
            emb, 
            embedding=True, 
            eof=eofs.cuda()
        )
    )

def get_gpt_centroids(gpt_descriptions, classnames, model, tokenizer,use_raw=False):
    '''
    gpt_descriptions is a dict where keys are classnames
    and entries are lists with input texts.
    Push each entry through the text encoder and calculate the 
    centroids.
    '''
    text_centroids = []
    for c in classnames:
        classname = transform_classname(replace_underscores(c))
        #print("classname,",classname)
        _descriptions = gpt_descriptions[classname]
        #print("_descriptions,",_descriptions)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            text_centroids.append(
                F.normalize(encode_text_wrapper(
                    model, _descriptions, tokenizer,use_raw
                )).mean(0)
            )
    text_centroids = torch.stack(text_centroids)
    text_centroids = F.normalize(text_centroids)
    return text_centroids

def get_centroids(closest_description_dict, 
                  classnames, model, tokenizer):
    '''
    gpt_descriptions is a dict where keys are classnames
    and entries are lists with input texts.
    Push each entry through the text encoder and calculate the 
    centroids.
    '''
    text_centroids = []
    for c in classnames:
        assert c in closest_description_dict
        assert c[-1] == '.'
        assert all([desc[0] == ' ' for desc in closest_description_dict[c]])
        _descriptions = [c[:-1] + ',' + desc for desc in closest_description_dict[c]]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            text_centroids.append(
                F.normalize(encode_text_wrapper(
                    model, _descriptions, tokenizer,use_raw=False
                )).mean(0)
            )
    text_centroids = torch.stack(text_centroids)
    text_centroids = F.normalize(text_centroids)
    return text_centroids
