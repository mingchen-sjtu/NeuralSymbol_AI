import os
 
class BatchRename():
 
    def rename(self):
        path = "/home/ur/Desktop/attribute_infer/bolt/data-end2end-triple/true_mul_bolt_crops/star_bolt"
        filelist = os.listdir(path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(path), item)
                dst = os.path.join(os.path.abspath(path), ''+str(i).zfill(3)+'.jpg')
                try:
                    os.rename(src, dst)
                    i += 1
                except:
                    continue
        print('total %d to rename & converted %d jpg'%(total_num, i))
 
if __name__=='__main__':
    demo = BatchRename()
    demo.rename()