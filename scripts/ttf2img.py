import os
import pygame
import multiprocessing
import json

def ttf2img(ttf_root, characters, img_dir):
    
    pygame.init()
    os.makedirs(img_dir, exist_ok=True)

    ttf_list = os.listdir(ttf_root)
    ttf_list.sort()
    
    ttf_number = len(ttf_list)
    ttf_root_list = [ttf_root]*ttf_number
    img_dir_list = [img_dir]*ttf_number
    characters_list = [characters]*ttf_number
    args = [(ttf_list[i],ttf_root_list[i],img_dir_list[i],characters_list[i]) for i in range(ttf_number)]
    with multiprocessing.Pool() as pool:
        pool.map(fonts_generate,args)

    print("ttf2img OK")

def fonts_generate(args):
    
    (ttf,ttf_root,img_dir,characters) = args
    fonts = []
    ttf_file = ttf_root + '/' + ttf
    ttf_name = ttf.split('.')[0]
    
    save_img_dir = img_dir + '/' + ttf_name
    os.makedirs(save_img_dir, exist_ok=True)


    for zone in range(72):
        zone_data = 0xB0A0 + zone * 0x0100

        for pos in range(6*16):

            if pos in [0, 6*16-1]:
                continue
            pos_data = zone_data + pos

            if pos_data in [0xD7FA, 0xD7FB, 0xD7FC, 0xD7FD, 0xD7FE]:
                continue
            pos_data = pos_data.to_bytes(2, byteorder='big')

            font = pos_data.decode('gbk')
            fonts.append(font)

    if True:

        for zone in range(32):
            zone_data = 0x8140 + zone * 0x0100

            for pos in range(12*16):
                
                if pos in [3*16+15, 11*16+15]:
                    continue
                pos_data = zone_data + pos
                pos_data = pos_data.to_bytes(2, byteorder='big')
                
                font = pos_data.decode('gbk')
                fonts.append(font)


        for zone in range(85):
            zone_data = 0xAA40 + zone * 0x0100

            for pos in range(6*16+1):
                
                if pos in [3*16+15]:
                    continue
                pos_data = zone_data + pos

                if pos_data >= 0xFE50:
                    continue
                pos_data = pos_data.to_bytes(2, byteorder='big')
                
                font = pos_data.decode('gbk')
                fonts.append(font)
            
    crop_size = 256
    font_render = pygame.font.Font(ttf_file, crop_size)
    k = 0
    for character in fonts:
        if character in characters:
            character_render = font_render.render(character, True, (0,0,0), (255,255,255))
            character_img = pygame.transform.scale(character_render, (crop_size, crop_size))
            pygame.image.save(character_img, os.path.join(save_img_dir, '{}.png'.format(character)))
            k = k+1
            print(f'\rRenderring font {ttf:8},length{k:5}:{len(characters):5}', end='')

if __name__ == '__main__':
    ttf_root = 'datasets/CFG/ttf_font_extra'
    with open ('datasets/CFG/chars_800.json', 'r') as f:
        characters = json.load(f)
    img_dir = 'datasets/CFG/font_extra_800'
    ttf2img(ttf_root, characters, img_dir)