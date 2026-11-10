import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os

from vit_gcn import ViT,GCN
from functions import normalize,get_data,GET_A2,metrics,show_results,train_and_test_data,train_epoch,valid_epoch,output_metric,applyPCA,gain_neighborhood_pixel
from sklearn.metrics import confusion_matrix
import run_utils as ru


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Salinas'],
                    default='Indian', help='dataset to use')

parser.add_argument("--num_run", type=int, default=5)
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')#奇数 9
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=70, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')#64
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
# --- Eklenen parametreler (test sızıntısız seçim ve kayıt) ---
parser.add_argument('--val_per_class', type=int, default=10,
                    help='sınıf başına en fazla doğrulama pikseli (eğitim dışı piksellerden)')
parser.add_argument('--out_dir', default='',
                    help='sonuçların yazılacağı klasör (boşsa kayıt yapılmaz)')
parser.add_argument('--wandb_project', default='',
                    help='W&B proje adı (boşsa W&B kullanılmaz)')
parser.add_argument('--wandb_group', default='', help='W&B grup adı')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
timer = ru.Timer()
# -------------------------------------------------------------------------------

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data

input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true = get_data(args.dataset)
##########得到原始图像 训练测试以及所有点坐标 每一类训练测试的个数############

# --- Eklenen: test havuzundan doğrulama kümesi ayır (eğitim kümesi değişmez) ---
total_pos_val, y_val, total_pos_test, y_test, val_per_class = ru.carve_validation(
    total_pos_test, y_test, num_classes, max_per_class=args.val_per_class)
train_per_class = np.bincount(y_train, minlength=num_classes).tolist()
test_per_class = np.bincount(y_test, minlength=num_classes).tolist()
print("train={0}, val={1}, test={2}".format(len(y_train), len(y_val), len(y_test)))


input = applyPCA(input, numComponents=args.pca_band)
# normalize data by band norm
input_normalize = normalize(input)
height, width, band = input_normalize.shape  # 145*145*200
print("height={0},width={1},band={2}".format(height, width, band))
# -------------------------------------------------------------------------------
# obtain train and test data
x_train_band, x_test_band, x_true_band, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_ture = train_and_test_data(
    input_normalize, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, w=height, h=width,
    n_gcn=args.n_gcn)
##########得到训练测试以及所有点的光谱############

# --- Eklenen: doğrulama pikselleri için aynı yama/komşu çıkarımı ---
x_val_band = np.zeros((total_pos_val.shape[0], args.patches, args.patches, band), dtype=float)
corner_val = np.zeros((total_pos_val.shape[0], 2), dtype=int)
indexs_val = torch.zeros((total_pos_val.shape[0], args.n_gcn), dtype=int).cuda()
for j in range(total_pos_val.shape[0]):
    x_val_band[j, :, :, :], corner_val[j, :], indexs_val[j] = gain_neighborhood_pixel(
        input_normalize, total_pos_val, j, args.patches, height, width, args.n_gcn)
print("x_val   shape = {}".format(x_val_band.shape))



input2 = torch.from_numpy(input_normalize).type(torch.FloatTensor)

A_train = GET_A2(x_train_band, input2, corner=corner_train,patches=args.patches , l=3,sigma=10)
x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)  # [695, 200, 7, 7]
y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
Label_train = Data.TensorDataset(A_train, x_train, y_train)

A_test = GET_A2(x_test_band, input2, corner=corner_test, patches=args.patches ,l=3, sigma=10)
x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)  # [9671, 200, 7, 7]
y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
Label_test = Data.TensorDataset(A_test, x_test, y_test)

A_val = GET_A2(x_val_band, input2, corner=corner_val, patches=args.patches, l=3, sigma=10)
x_val = torch.from_numpy(x_val_band).type(torch.FloatTensor)
y_val = torch.from_numpy(y_val).type(torch.LongTensor)
Label_val = Data.TensorDataset(A_val, x_val, y_val)

x_true = torch.from_numpy(x_true_band).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
##########训练集的光谱值及标签##########
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
##########测试集的光谱值及标签##########
# Doğrulama yükleyicisi test yükleyicisiyle aynı şekilde kurulur (shuffle dahil),
# böylece iki ölçüm aynı değerlendirme yolundan geçer.
label_val_loader = Data.DataLoader(Label_val, batch_size=args.batch_size, shuffle=True)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)
##########所有地物的光谱值及标签##########

# --- Eklenen: koşu kaydı ---
git = ru.git_info(REPO_DIR)
config = dict(vars(args))
config.update({
    "model": "GTFN",
    "protocol": "original_split + val_selection",
    "git": git,
    "env": ru.env_info(),
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "train_per_class": train_per_class, "val_per_class_counts": val_per_class,
    "test_per_class": test_per_class,
})
if args.out_dir:
    os.makedirs(args.out_dir, exist_ok=True)
    ru.save_json(os.path.join(args.out_dir, "config.json"), config)
wb = ru.WandbLogger(
    args.wandb_project, config,
    name="{}_seed{:02d}".format(args.dataset, args.seed),
    group=args.wandb_group or "GTFN_{}".format(args.dataset),
    tags=["GTFN", args.dataset, "noleak", "original-split"],
    run_dir=None)  # W&B'nin kendi dosyaları Drive'a değil, yerel wandb klasörüne yazılır


results = []

best_OA2 = 0.0
best_AA_mean2 = 0.0
best_Kappa2 = 0.0
gcn_net = GCN(height, width, band, num_classes)
gcn_net = gcn_net.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(gcn_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//2, gamma=args.gamma)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
# -------------------------------------------------------------------------------

tr_net = ViT(
    n_gcn=args.n_gcn,
    num_patches=64,
    num_classes=num_classes,
    dim=64,
    depth=5,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
)
tr_net = tr_net.cuda()
# optimizer
optimizer2 = torch.optim.Adam(tr_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.epoches // 10, gamma=args.gamma)


def evaluate(loader, indexs):
    """Orijinal valid_epoch ile değerlendirme; gradyan tutulmaz (sonuç aynı)."""
    with torch.no_grad():
        tar, pre = valid_epoch(gcn_net, tr_net, loader, criterion, indexs)
    OA, AA_mean, Kappa, AA = output_metric(tar, pre)
    return {"OA": OA * 100.0, "AA": AA_mean * 100.0, "Kappa": Kappa * 100.0,
            "class acc": AA * 100.0, "tar": tar, "pre": pre}


def better(cur, best):
    """Orijinal koddaki seçim kuralı: OA, AA ve Kappa'nın üçü de >= olmalı."""
    if best is None:
        return True
    return cur["OA"] >= best["OA"] and cur["AA"] >= best["AA"] and cur["Kappa"] >= best["Kappa"]


best_val = None      # doğrulamaya göre seçilen epoch (RAPORLANACAK SONUÇ)
sel_test = None      # o epoch'taki test sonucu
oracle_test = None   # orijinal kural: teste göre seçim (yalnızca sızıntıyı ölçmek için)
last_test = None     # son epoch'taki test sonucu
history = []
train_time = 0.0
eval_time = 0.0
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("start training")
tic = time.time()
for epoch in range(args.epoches):
    scheduler.step()
    scheduler2.step()
    # train model
    gcn_net.train()
    tr_net.train()
    t_ep = time.time()
    train_acc, train_obj, tar_t, pre_t = train_epoch(gcn_net, tr_net, label_train_loader, criterion, optimizer,
                                                     optimizer2, indexs_train)
    train_time += time.time() - t_ep
    OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
    print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
          .format(epoch + 1, train_obj, train_acc))
    row = {"epoch": epoch + 1, "train_loss": float(train_obj), "train_acc": float(train_acc),
           "lr": optimizer.param_groups[0]["lr"]}

    # Orijinal değerlendirme takvimi (operatör önceliği korunarak açık yazıldı):
    is_eval = ((epoch % args.test_freq == 0) or (epoch == args.epoches - 1)) and epoch >= args.epoches * 0.6
    if is_eval:
        t_ev = time.time()
        gcn_net.eval()
        tr_net.eval()
        # 1) Test: orijinal kodla aynı sırada ve aynı şekilde
        te = evaluate(label_test_loader, indexs_test)
        # 2) Doğrulama: RNG durumu korunur, böylece eğitimin geri kalanı
        #    orijinal koşudaki rastgele sırayı izler.
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        va = evaluate(label_val_loader, indexs_val)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        eval_time += time.time() - t_ev

        te["epoch"] = va["epoch"] = epoch + 1
        if better(va, best_val):
            best_val, sel_test = va, te
        if better(te, oracle_test):
            oracle_test = te
        last_test = te
        print("   eval epoch {:03d}: val OA={:.2f}  test OA={:.2f}  (seçili epoch={})"
              .format(epoch + 1, va["OA"], te["OA"], best_val["epoch"]))
        row.update({"val_OA": va["OA"], "val_AA": va["AA"], "val_Kappa": va["Kappa"],
                    "test_OA": te["OA"], "test_AA": te["AA"], "test_Kappa": te["Kappa"]})
    history.append(row)
    wb.log(row, step=epoch + 1)

toc = time.time()

# Orijinal biçimde rapor: artık doğrulamaya göre seçilen epoch'un test sonucu
run_results = metrics(sel_test["OA"] / 100.0, sel_test["AA"] / 100.0,
                      sel_test["Kappa"] / 100.0, sel_test["class acc"] / 100.0)
show_results(run_results,agregated=False)
results.append(run_results)


def _summ(d):
    return {"OA": d["OA"], "AA": d["AA"], "Kappa": d["Kappa"],
            "class_acc": d["class acc"], "epoch": d["epoch"]}


final = {
    "dataset": args.dataset, "seed": args.seed,
    "selected": _summ(sel_test),             # raporlanacak sonuç
    "selected_val": _summ(best_val),
    "last_epoch": _summ(last_test),
    "test_oracle_leaky": _summ(oracle_test),  # orijinal (sızıntılı) yöntem
    "leak_gap_OA": oracle_test["OA"] - sel_test["OA"],
    "time_total_s": timer.elapsed(),
    "time_train_s": train_time,
    "time_eval_s": eval_time,
    "time_per_epoch_s": train_time / max(args.epoches, 1),
    "peak_gpu_mem_MB": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
    "git": git,
}
print("SEÇİLEN (val) epoch={} test OA={:.2f} | son epoch OA={:.2f} | sızıntılı OA={:.2f} (fark {:+.2f})".format(
    sel_test["epoch"], sel_test["OA"], last_test["OA"], oracle_test["OA"], final["leak_gap_OA"]))

if args.out_dir:
    ru.save_history(os.path.join(args.out_dir, "history.csv"), history)
    np.save(os.path.join(args.out_dir, "confusion_test.npy"),
            confusion_matrix(sel_test["tar"], sel_test["pre"], labels=list(range(num_classes))))
    ru.save_json(os.path.join(args.out_dir, "metrics.json"), final)  # en son yazılır = koşu tamam

wb.summary({
    "selected/test_OA": sel_test["OA"], "selected/test_AA": sel_test["AA"],
    "selected/test_Kappa": sel_test["Kappa"], "selected/epoch": sel_test["epoch"],
    "selected/val_OA": best_val["OA"],
    "last/test_OA": last_test["OA"],
    "leaky/test_OA": oracle_test["OA"], "leaky/epoch": oracle_test["epoch"],
    "leak_gap_OA": final["leak_gap_OA"],
    "time_total_s": final["time_total_s"], "time_per_epoch_s": final["time_per_epoch_s"],
    "peak_gpu_mem_MB": final["peak_gpu_mem_MB"],
})
wb.finish()
