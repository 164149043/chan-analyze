#!/usr/bin/env python
# coding: utf-8
"""
增量版缠论核心（cl_core_inc.py）——按 2024-12-27 逆向的增量状态机实现
============================================================================
process_klines 是逐根增量：每根新K线依次调用
  _klines_baohan([k], cl_klines)  →  process_fx()  →  process_bi()
分型（done 候选/确认 + 位置漂移）、笔（最后笔延伸/新笔配对）都是增量演化的。
全量重算无法复现，本模块按逆向翻译（reverse/CL2024_reverse.md）逐根实现。

逆向关键点：
- _create_fx：严格不等号；3根确认(done=True)/2根候选(done=False)；单调序列用k3
- process_fx：滑动窗口 [start, start+2]，fxs 引用更新 + 增量修正（删除/重算）
- process_bi：
  * bis 空：主循环从 fxs[0] 配对（同向 val 替换起点），成笔即返回
  * bis 非空：826 验证最后笔（失效删除重算）
             832 配对：最后笔 end 之后"与 end 反向"分型，反转取最远可行 → 新笔
             869 延伸：最后笔 start 与"与 end 反向"分型，反转取最远可行 → end 延伸
  * 869 filter 是与 end 反向（不是 start！）——这是 4→64 锁定的关键
- _bi_check_bi_fx_ok：(A)互包含+k_index间隔<13 (B)方向极值 (C)strict原始K线极值 (F1)index≥4
"""
import datetime

import numpy as np
import talib as ta


# ==================== 基础对象 ====================
class CK:  # 缠论K线（合并后）
    __slots__ = ("index", "k_index", "h", "l", "o", "c", "a", "n", "ks", "up_qs", "date")

    def __init__(self, index, k_index, h, l, o, c, a, date):
        self.index = index
        self.k_index = k_index
        self.h = h
        self.l = l
        self.o = o
        self.c = c
        self.a = a
        self.n = 1
        self.ks = []            # 包含的原始K线 (idx, k_dict)
        self.up_qs = None
        self.date = date


class FX:
    __slots__ = ("type", "k", "klines", "index", "done", "k_index", "real")

    def __init__(self, _type, k, klines, val, index=0, done=True):
        self.type = _type       # ding / di
        self.k = k              # 中间缠论K线
        self.klines = klines    # 三根缠论K线
        self.index = index      # 分型序号
        self.done = done
        self.k_index = k.k_index
        self.real = True

    @property
    def val(self):
        """分型值 = 中间K线的 h/l（动态属性，漂移时跟随 k 引用更新）"""
        return self.k.h if self.type == "ding" else self.k.l

    def high(self):
        hs = [k["h"] for ck in self.klines if ck is not None for _, k in ck.ks]
        return max(hs) if hs else self.val

    def low(self):
        ls = [k["l"] for ck in self.klines if ck is not None for _, k in ck.ks]
        return min(ls) if ls else self.val


class MMD:
    """买卖点对象（cl_interface.py 定义）"""
    __slots__ = ("name", "zs", "msg")

    def __init__(self, name, zs):
        self.name = name
        self.zs = zs
        self.msg = ""

    def __str__(self):
        return f"MMD: {self.name} ZS: {self.zs} MSG: {self.msg}"


class BC:
    """背驰对象（cl_interface.py 定义）"""
    __slots__ = ("type", "zs", "compare_line", "compare_lines", "bc")

    def __init__(self, _type, zs, compare_line, compare_lines, bc):
        self.type = _type
        self.zs = zs
        self.compare_line = compare_line
        self.compare_lines = compare_lines
        self.bc = bc

    def __str__(self):
        return f"BC type: {self.type} bc: {self.bc} zs: {self.zs}"


class BI:
    __slots__ = ("index", "start", "end", "high", "low", "type", "fx_num", "done",
                 "ld", "mmds", "bcs", "zs_type_mmds", "zs_type_bcs",
                 "default_zs_type", "qs_beichi", "pz_beichi", "is_split",
                 "zs_high", "zs_low", "start_line", "end_line")

    def __init__(self, index, start_fx, end_fx=None, _type=None, default_zs_type="zs_type_bz"):
        self.index = index
        self.start = start_fx
        self.end = end_fx
        self.type = _type
        if end_fx is not None:
            self.high = max(start_fx.val, end_fx.val)
            self.low = min(start_fx.val, end_fx.val)
        else:
            self.high = None
            self.low = None
        self.fx_num = 0
        self.done = True
        self.ld = None
        self.mmds = []
        self.bcs = []
        self.zs_type_mmds = {}
        self.zs_type_bcs = {}
        self.default_zs_type = default_zs_type
        self.qs_beichi = None
        self.pz_beichi = None
        self.is_split = ""
        self.zs_high = None
        self.zs_low = None
        self.start_line = None
        self.end_line = None

    def ding_high(self):
        return self.end.val if self.type == "up" else self.start.val

    def di_low(self):
        return self.end.val if self.type == "down" else self.start.val

    def is_done(self):
        return self.end.done

    def __str__(self):
        return f"index: {self.index} type: {self.type} FX: ({self.start.k.date} - {self.end.k.date}) " \
               f"high: {self.high} low: {self.low} done: {self.is_done()}"


CONFIG = {
    "kline_type": "kline_default",
    "kline_qk": "none",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "fx_bh": "fx_bh_yes",
    "bi_type": "bi_type_old",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "bi_qj": "bi_qj_dd",
    "fx_check_k_nums": 13,
    "allow_bi_fx_strict": True,     # CL() 默认 True（(C) 生效）
    "bi_bzh": "bi_bzh_yes",
    "bi_split_k_cross_nums": "20,1",
    "xd_allow_bi_pohuai": "yes",    # 默认支持笔破坏
    "xd_qj": "xd_qj_dd",
    "zsd_qj": "zsd_qj_dd",
    "zs_qj": "zs_qj_dd",
    "xd_allow_split_no_highlow": True,
    "xd_allow_split_zs_kz": False,
    "xd_allow_split_zs_more_line": True,
    "xd_allow_split_zs_no_direction": True,
    "xd_zs_max_lines_split": 11,
    "zs_optimize": False,
    "zs_bi_type": ["zs_type_bz"],
    "zs_xd_type": ["zs_type_bz"],
    "zs_wzgx": "zs_wzgx_zggdd",     # CL() 默认 zggdd（趋势判断用）
    "zs_cd": "zs_cd_three",         # 中枢重叠：前三段
    "judge_zs_qs_level": 1,         # 趋势中枢级别判断
    "idx_macd_fast": 12,
    "idx_macd_slow": 26,
    "idx_macd_signal": 9,
    "cl_mmd_cal_qs_1mmd": 1,
    "cl_mmd_cal_not_qs_3mmd_1mmd": 1,
    "cl_mmd_cal_qs_3mmd_1mmd": 1,
    "cl_mmd_cal_qs_not_lh_2mmd": 1,
    "cl_mmd_cal_qs_bc_2mmd": 1,
    "cl_mmd_cal_3mmd_not_lh_bc_2mmd": 1,
    "cl_mmd_cal_1mmd_not_lh_2mmd": 1,
    "cl_mmd_cal_3mmd_xgxd_not_bc_2mmd": 1,
    "cl_mmd_cal_not_in_zs_3mmd": 1,
    "cl_mmd_cal_not_in_zs_gt_9_3mmd": 1,
    "__cal_xd": 1,
    "__cal_xd_youhua": True,
    "__cal_xd_zss": 1,
    "__cal_xd_mmd": 1,
    "__cal_zsd": 1,
    "__cal_zsd_zss": 1,
    "__cal_zsd_mmd": 1,
    "__cal_qsd": 1,
    "__cal_qsd_zss": 1,
    "__cal_qsd_mmd": 1,
}


class CL:
    def __init__(self, code=None, frequency=None, config=None):
        self.code = code if code is not None else "000533"
        self.frequency = frequency if frequency is not None else "30m"
        self.config = dict(CONFIG)
        if config:
            for k, v in config.items():
                if k in self.config:
                    self.config[k] = v
                else:
                    self.config[k] = v
                if "cl_mmd_cal" in k or k.startswith("xd_allow_split_"):
                    self.config[k] = bool(int(self.config[k]))
            # 类型转换（CL.__init__ 179-199 行）
            if isinstance(self.config.get("zs_bi_type"), str):
                self.config["zs_bi_type"] = [self.config["zs_bi_type"]]
            if isinstance(self.config.get("zs_xd_type"), str):
                self.config["zs_xd_type"] = [self.config["zs_xd_type"]]
            if isinstance(self.config.get("xd_zs_max_lines_split"), str):
                self.config["xd_zs_max_lines_split"] = int(self.config["xd_zs_max_lines_split"])
            if isinstance(self.config.get("fx_check_k_nums"), str):
                self.config["fx_check_k_nums"] = int(self.config["fx_check_k_nums"])
            for k in ("idx_macd_fast", "idx_macd_slow", "idx_macd_signal"):
                if isinstance(self.config.get(k), str):
                    self.config[k] = int(self.config[k])
            for k in ("allow_bi_fx_strict", "zs_optimize", "judge_zs_qs_level"):
                if isinstance(self.config.get(k), str):
                    self.config[k] = bool(int(self.config[k]))
        self.src_klines = []     # 原始K线 (k dict)
        self.cl_klines = []      # 缠论K线（合并后）
        self.fxs = []            # 分型（含 done=False 候选）
        self.bis = []            # 笔
        self.xds = []            # 线段
        self.zsds = []           # 走势段
        self.qsds = []           # 趋势段
        self.bi_zss = []         # 笔中枢
        self.xd_zss = []         # 线段中枢
        self.type_bi_zss = {}    # 类型 -> 笔中枢列表
        self.type_xd_zss = {}    # 类型 -> 线段中枢列表
        self.type_zsd_zss = {}   # 类型 -> 走势段中枢列表
        self.qsd_zss = []        # 趋势段中枢
        self.idx = {"macd": {"dea": [], "dif": [], "hist": []}}
        self._CL__last_up_line_end_ukey = {}
        self._CL__last_bi_end_ukey = None
        self.k_index = 0         # 原始K线计数

    # ---------- K线包含合并（_klines_baohan：批量合并原始K线） ----------
    def _klines_baohan(self, klines, up_cl_klines=None):
        """合并原始K线列表进缠论K线（ 批量版，src K 带 index 序号）"""
        up_cl = self.cl_klines if up_cl_klines is None else up_cl_klines
        for k in klines:
            up_kline = up_cl[-1]
            # 方向
            if len(up_cl) >= 2:
                qushi = "up" if up_cl[-2].h < up_cl[-1].h else "down"
            else:
                qushi = "up" if up_cl[-1].c > up_cl[-1].o else "down"
            # 包含判断（创建新对象替换 cl_klines[-1]，旧对象供分型引用）
            if (up_kline.h >= k["h"] and up_kline.l <= k["l"]) or \
               (k["h"] >= up_kline.h and k["l"] <= up_kline.l):
                if qushi == "up":
                    new_h = max(up_kline.h, k["h"])
                    new_l = max(up_kline.l, k["l"])
                    new_kidx = k["index"] if k["h"] > up_kline.h else up_kline.k_index
                    new_date = k["t"] if k["h"] > up_kline.h else up_kline.date
                    new_c, new_o = new_h, new_l
                else:
                    new_h = min(up_kline.h, k["h"])
                    new_l = min(up_kline.l, k["l"])
                    new_kidx = k["index"] if k["l"] < up_kline.l else up_kline.k_index
                    new_date = k["t"] if k["l"] < up_kline.l else up_kline.date
                    new_c, new_o = new_l, new_h
                new_ck = CK(up_kline.index, new_kidx, new_h, new_l, new_o, new_c,
                            up_kline.a + k["v"], new_date)
                new_ck.ks = list(up_kline.ks) + [(k["index"], k)]
                new_ck.n = up_kline.n + 1
                new_ck.up_qs = "up" if qushi == "up" else "down"
                up_cl[-1] = new_ck
            else:
                cur = CK(up_kline.index + 1, k["index"], k["h"], k["l"],
                         k["l"] if k["c"] > k["o"] else k["h"],
                         k["h"] if k["c"] > k["o"] else k["l"], k["v"], k["t"])
                cur.ks.append((k["index"], k))
                up_cl.append(cur)

    def process_cl_kline(self):
        """合并最新原始K线（ process_cl_kline：删最后2根缠论K线后批量重放）"""
        v4 = self.src_klines
        if len(v4) == 0:
            return False
        if len(self.cl_klines) == 0:
            v15 = v4[0]
            if v15["c"] > v15["o"]:
                v32 = CK(v15["index"], 0, v15["h"], v15["l"], v15["l"], v15["h"],
                         v15["v"], v15["t"])
            else:
                v32 = CK(v15["index"], 0, v15["h"], v15["l"], v15["h"], v15["l"],
                         v15["v"], v15["t"])
            v32.ks.append((v15["index"], v15))
            self.cl_klines.append(v32)
        while len(self.cl_klines) >= 3:
            del self.cl_klines[-1]
            del self.cl_klines[-1]
        while len(self.cl_klines) > 0 and self.cl_klines[-1].k_index == -1:
            del self.cl_klines[-1]
        v33 = [k for k in reversed(v4) if k["index"] > self.cl_klines[-1].ks[-1][0]]
        v33 = list(reversed(v33))   # 从尾部收集后反转回原序（v4[::-1] 过滤后 [::-1]）
        self._klines_baohan(v33)
        return True

    # ---------- 分型创建（_create_fx） ----------
    def _create_fx(self, klines):
        if len(klines) == 3:
            k1, k2, k3 = klines[0], klines[1], klines[2]
        elif len(klines) == 2:
            k1, k2, k3 = klines[0], klines[1], None
        else:
            return None
        if k3 is not None:
            if k1.h > k2.h and k3.h > k2.h and k2.l < k1.l and k2.l < k3.l:
                return FX("di", k2, klines, k2.l, done=True)
            if k1.l < k2.l and k3.l < k2.l and k2.h > k1.h and k2.h > k3.h:
                return FX("ding", k2, klines, k2.h, done=True)
            if k3.h < k2.h and k2.h < k1.h and k3.l < k2.l and k2.l < k1.l:
                return FX("di", k3, [k3, k2, k3], k3.l, done=False)
            if k3.l > k2.l and k2.l > k1.l and k3.h > k2.h and k2.h > k1.h:
                return FX("ding", k3, [k3, k2, k3], k3.h, done=False)
        if k3 is None:
            if k2.h < k1.h and k2.l < k1.l:
                return FX("di", k2, [k1, k2], k2.l, done=False)
            if k2.h > k1.h and k2.l > k1.l:
                return FX("ding", k2, [k1, k2], k2.h, done=False)
        return None

    # ---------- 分型增量（process_fx） ----------
    def process_fx(self):
        if len(self.cl_klines) < 3:
            return False
        fxs = self.fxs
        # 增量修正已有分型
        if len(fxs) > 0:
            last_fx = fxs[-1]
            last_k = last_fx.klines[-1] if last_fx.klines[-1] is not None else last_fx.klines[1]
            if last_fx.done is False and last_k.index == self.cl_klines[-1].index \
               and last_k.h == self.cl_klines[-1].h and last_k.l == self.cl_klines[-1].l:
                return False
            if last_k.index > self.cl_klines[-1].index:
                del fxs[last_fx.index]
                if len(fxs) == 0:
                    return True
            elif last_fx.done and (self.cl_klines[last_fx.klines[-1].index].h != last_fx.klines[-1].h
                                   or self.cl_klines[last_fx.klines[-1].index].l != last_fx.klines[-1].l):
                del fxs[last_fx.index]
            elif last_fx.done is False:
                del fxs[last_fx.index]
            if len(fxs) == 0:
                return True
        # 更新 fxs 引用
        for fx in fxs[-5:]:
            fx.k = self.cl_klines[fx.k.index]
            for i in range(len(fx.klines)):
                if fx.klines[i] is not None:
                    if fx.klines[i].index > self.cl_klines[-1].index:
                        del fxs[-1]
                        break
                    fx.klines[i] = self.cl_klines[fx.klines[i].index]
        # 滑动窗口扫描
        has_new = False
        start_idx = fxs[-1].k.index if len(fxs) > 0 else 0
        end_idx = min(start_idx + 2, self.cl_klines[-1].index)
        last_idx = self.cl_klines[-1].index
        while True:
            seg = self.cl_klines[start_idx: end_idx + 1]
            if len(seg) < 2:
                break
            fx = self._create_fx(seg)
            if fx is None:
                start_idx += 1
                end_idx = min(end_idx + 1, last_idx)
                continue
            if fx.done:
                fx.index = fxs[-1].index + 1 if len(fxs) > 0 else 0
                fxs.append(fx)
                has_new = True
                start_idx = fx.k.index
                end_idx = min(start_idx + 2, last_idx)
            else:
                if end_idx == last_idx:
                    fx.index = fxs[-1].index + 1 if len(fxs) > 0 else 0
                    fxs.append(fx)
                    has_new = True
                    break
                else:
                    start_idx += 1
                    end_idx = min(end_idx + 1, last_idx)
        return has_new

    # ---------- 笔确认（_bi_check_bi_fx_ok） ----------
    def _bi_check_bi_fx_ok(self, start_fx, end_fx):
        if start_fx.type == end_fx.type:
            return False
        fx_check_k_nums = self.config.get("fx_check_k_nums", 13)
        bi_type = self.config.get("bi_type", "bi_type_old")
        if bi_type in ("bi_type_old", "bi_type_new"):
            if end_fx.k.k_index - start_fx.k.k_index < fx_check_k_nums:
                if (start_fx.k.h > end_fx.k.h and start_fx.k.l < end_fx.k.l) or \
                   (start_fx.k.h < end_fx.k.h and start_fx.k.l > end_fx.k.l):
                    return False
        if start_fx.type == "ding" and start_fx.k.h < end_fx.k.l:
            return False
        if start_fx.type == "di" and start_fx.k.l > end_fx.k.h:
            return False
        if self.config.get("allow_bi_fx_strict") is True:
            if end_fx.k.k_index - start_fx.k.k_index < fx_check_k_nums:
                if start_fx.type == "ding" and start_fx.low() < end_fx.low():
                    return False
                if start_fx.type == "di" and start_fx.high() > end_fx.high():
                    return False
                if end_fx.type == "ding" and end_fx.low() < start_fx.low():
                    return False
                if end_fx.type == "di" and end_fx.high() > start_fx.high():
                    return False
        if bi_type == "bi_type_old":
            if end_fx.k.index - start_fx.k.index < 4:
                return False
        if bi_type == "bi_type_new":
            if end_fx.k.index - start_fx.k.index < 3:
                return False
            start_kl = max(start_fx.k.ks, key=lambda x: x[1]["h"]) \
                if start_fx.type == "ding" else min(start_fx.k.ks, key=lambda x: x[1]["l"])
            end_kl = max(end_fx.k.ks, key=lambda x: x[1]["h"]) \
                if end_fx.type == "ding" else min(end_fx.k.ks, key=lambda x: x[1]["l"])
            if end_kl[0] - start_kl[0] < 4:
                return False
        return True

    # ---------- 笔增量（process_bi） ----------
    def _mk_bi(self, start_fx, end_fx):
        bi = BI(len(self.bis), start_fx, end_fx, None)
        bi.type = "up" if start_fx.type == "di" else "down"
        bi.high = max(start_fx.val, end_fx.val)
        bi.low = min(start_fx.val, end_fx.val)
        return bi

    def process_bi(self):
        fxs = self.fxs
        if len(fxs) < 2:
            return False
        if len(self.bis) == 0:
            # 主循环：从 fxs[0] 配对（同向 val 替换起点）
            start_fx = fxs[0]
            for end_fx in fxs[1:]:
                if start_fx.type == "ding" and end_fx.type == "ding" and end_fx.val > start_fx.val:
                    start_fx = end_fx
                    continue
                if start_fx.type == "di" and end_fx.type == "di" and end_fx.val < start_fx.val:
                    start_fx = end_fx
                    continue
                if end_fx.index > start_fx.index and end_fx.type != start_fx.type:
                    if self._bi_check_bi_fx_ok(start_fx, end_fx):
                        self.bis.append(self._mk_bi(start_fx, end_fx))
                        return True
            return False
        # 750 增量（bis 非空）
        # 823+826：更新最后笔 end 引用 + 826 验证（失败删除后直接进重算循环）
        if self.bis[-1].end.index <= fxs[-1].index:
            self.bis[-1].end = fxs[self.bis[-1].end.index]
        if not self._bi_check_bi_fx_ok(self.bis[-1].start, self.bis[-1].end):
            del self.bis[-1]
            if len(self.bis) == 0:
                return False
        # 重算循环：832 配对 + 869 延伸 + 标准拆分 + 特殊拆分，任一成功则再来一轮
        v46 = 0
        while True:
            v46 += 1
            if v46 >= 100:
                break
            changed = False
            # 832 配对：从最后笔 end 之后，与 end 反向，反转最远可行 → 新笔
            last = self.bis[-1]
            cands = [fx for fx in fxs
                     if fx.index > last.end.index and fx.type != last.end.type]
            for ef in reversed(cands):
                if self._bi_check_bi_fx_ok(last.end, ef):
                    self.bis.append(self._mk_bi(last.end, ef))
                    changed = True
                    break
            # 869 延伸：与 end 同型且更极端的候选，反转（最远优先），check(start, fx) True 则 end=fx
            # filter: fxs[end.index:] 中 index >= end.index 且 type == end.type
            #         且 (end.type=='ding' and fx.val > end.val) or (end.type=='di' and fx.val < end.val)
            last = self.bis[-1]
            end_fx = last.end
            cands2 = []
            for fx in fxs[end_fx.index:]:
                if fx.index >= end_fx.index and fx.type == end_fx.type:
                    if (end_fx.type == "ding" and fx.val > end_fx.val) or \
                       (end_fx.type == "di" and fx.val < end_fx.val):
                        cands2.append(fx)
            for ef in reversed(cands2):
                if self._bi_check_bi_fx_ok(last.start, ef) and ef.index > last.end.index:
                    last.end = ef
                    last.high = max(last.start.val, ef.val)
                    last.low = min(last.start.val, ef.val)
                    changed = True
                    break
            # 标准拆分：最后笔 start 不是笔内最极端时，
            # 把 bis[-2] 延伸到最极端分型，删除最后笔
            if self._bi_std_split():
                changed = True
            # 特殊拆分（重叠K线超阈值）
            if self._bi_special_bi_split(self.bis[-1]):
                changed = True
            if not changed:
                break
        # 收尾：最后笔 ukey 变化才返回 True
        if len(self.bis) == 0:
            return True
        ukey = self._line_ukey(self.bis[-1])
        if ukey != self._CL__last_bi_end_ukey:
            self._CL__last_bi_end_ukey = ukey
            return True
        return False

    def _bi_std_split(self):
        """标准拆分：
        down 笔：start.val != 笔内最高顶 → bis[-2].end = 笔内最高顶分型，del bis[-1]
        up 笔：start.val != 笔内最低底 → bis[-2].end = 笔内最低底分型，del bis[-1]"""
        fxs = self.fxs
        if len(self.bis) < 2:
            return False
        v42 = self.bis[-1]
        seg = fxs[v42.start.index: v42.end.index + 1]
        dings = [fx.val for fx in seg if fx.type == "ding"]
        dis_ = [fx.val for fx in seg if fx.type == "di"]
        if not dings or not dis_:
            return False
        v43 = max(dings)   # 笔内最高顶
        v44 = min(dis_)    # 笔内最低底
        if v42.type == "down" and v42.start.val != v43:
            v50 = [fx for fx in fxs[v42.start.index:]
                   if fx.index > v42.start.index and fx.type == v42.start.type
                   and fx.val == v43]
            v51 = self.bis[-2]
            if v51.is_split == "" and len(v50) > 0:
                if self._bi_check_bi_fx_ok(v51.start, v50[0]):
                    v51.end = v50[0]
                    self._process_line_hl(v51)
                    del self.bis[-1]
                    self.process_zs(["bi"])
                    self.process_mmd(["bi"])
                    return True
        elif v42.type == "up" and v42.start.val != v44:
            v50 = [fx for fx in fxs[v42.start.index:]
                   if fx.index > v42.start.index and fx.type == v42.start.type
                   and fx.val == v44]
            v51 = self.bis[-2]
            if v51.is_split == "" and len(v50) > 0:
                if self._bi_check_bi_fx_ok(v51.start, v50[0]):
                    v51.end = v50[0]
                    self._process_line_hl(v51)
                    del self.bis[-1]
                    self.process_zs(["bi"])
                    self.process_mmd(["bi"])
                    return True
        return False

    def _cross_qujian(self, qj_one, qj_two):
        """区间交叉（_cross_qujian）：返回 {'max','min'} 或 None"""
        _max = min(max(qj_two[0], qj_two[1]), max(qj_one[0], qj_one[1]))
        _min = max(min(qj_two[0], qj_two[1]), min(qj_one[0], qj_one[1]))
        if _max >= _min:
            return {"max": _max, "min": _min}
        return None

    def _bi_special_bi_split(self, bi):
        """特殊笔处理：重叠K线超过设置后进行拆分（ _bi_special_bi_split）"""
        cfg = self.config.get("bi_split_k_cross_nums", "20,1").split(",")
        try:
            v148 = max(13, int(cfg[0]))
            v149 = int(cfg[1])
        except (ValueError, IndexError):
            v148, v149 = 20, 1
        if v149 < 0 or v149 > 5:
            v149 = 1
        if v148 >= 99:
            return False
        v45 = self.fxs[bi.start.index + 1: bi.end.index]     # 笔内分型（不含端点）
        if len(v45) < 3:
            return False
        # 找重叠区域：连续 3 分型区间相交
        for i in range(len(v45) - 2):
            f1, f2, f3 = v45[i], v45[i + 1], v45[i + 2]
            cross = self._cross_qujian([f1.high(), f1.low()], [f2.high(), f2.low()])
            if cross is None:
                continue
            cross = self._cross_qujian([cross["max"], cross["min"]], [f3.high(), f3.low()])
            if cross is None:
                continue
            # 统计重叠区间内交叉K线数（src_klines[f1.k.k_index : bi.end.k.k_index]）
            v159, v160 = 0, 0
            for k in self.src_klines[f1.k.k_index: bi.end.k.k_index]:
                if self._cross_qujian([k["h"], k["l"]], [cross["max"], cross["min"]]) is None:
                    v159 += 1
                else:
                    v160 += 1
                    v159 = 0
                if v159 > v149:
                    break
            if v160 < v148:
                continue
            # 达到拆分条件
            v72 = sorted([fx for fx in v45 if fx.type == "ding"],
                         key=lambda f: f.val, reverse=True)   # 顶降序
            v71 = sorted([fx for fx in v45 if fx.type == "di"],
                         key=lambda f: f.val)                 # 底升序
            if bi.type == "up":
                for v162 in v72:
                    for v163 in v71:
                        if v163.index > v162.index and v163.val < v162.val \
                           and (v162.k.k_index - bi.start.k.k_index) > 3 \
                           and (v163.k.k_index - v162.k.k_index) > 3 \
                           and (bi.end.k.k_index - v163.k.k_index) > 3:
                            if not self._bi_check_bi_fx_ok(v163, bi.end):
                                continue
                            v168 = bi.end
                            bi.end = v162
                            bi.is_split = "split"
                            self._process_line_hl(bi)
                            self.process_zs(["bi"])
                            self.process_mmd(["bi"])
                            v169 = BI(len(self.bis), bi.end, v163, "down",
                                      self.default_xd_zs_type())
                            self._process_line_hl(v169)
                            self.process_zs(["bi"])
                            self.process_mmd(["bi"])
                            self.bis.append(v169)
                            v172 = BI(len(self.bis), v163, v168, "up",
                                      self.default_xd_zs_type())
                            self._process_line_hl(v172)
                            self.process_zs(["bi"])
                            self.process_mmd(["bi"])
                            self.bis.append(v172)
                            return True
            else:  # down 笔（对称）
                for v163 in v71:
                    for v162 in v72:
                        if v162.index > v163.index and v162.val > v163.val \
                           and (v163.k.k_index - bi.start.k.k_index) > 3 \
                           and (v162.k.k_index - v163.k.k_index) > 3 \
                           and (bi.end.k.k_index - v162.k.k_index) > 3:
                            if not self._bi_check_bi_fx_ok(v162, bi.end):
                                continue
                            v168 = bi.end
                            bi.end = v163
                            bi.is_split = "split"
                            self._process_line_hl(bi)
                            self.process_zs(["bi"])
                            self.process_mmd(["bi"])
                            v169 = BI(len(self.bis), bi.end, v162, "up",
                                      self.default_xd_zs_type())
                            self._process_line_hl(v169)
                            self.process_zs(["bi"])
                            self.process_mmd(["bi"])
                            self.bis.append(v169)
                            v172 = BI(len(self.bis), v162, v168, "down",
                                      self.default_xd_zs_type())
                            self._process_line_hl(v172)
                            self.process_zs(["bi"])
                            self.process_mmd(["bi"])
                            self.bis.append(v172)
                            return True
        return False

    def _process_line_hl(self, line):
        """计算笔的 high/low（_process_line_hl：端点极值）"""
        line.high = max(line.start.val, line.end.val)
        line.low = min(line.start.val, line.end.val)

    # ---------- 逐根驱动（ process_klines 调用链） ----------
    def process_klines(self, df):
        # 先删除最后一根已处理的K线（增量重放最后一根）
        if len(self.src_klines) > 0:
            del self.src_klines[-1]
        v13 = len(self.src_klines)
        if v13 == 0:
            v14 = df
        else:
            v14 = df[df["date"] >= self.src_klines[-1]["t"]]
        for _, row in v14.iterrows():
            k = KDict(h=float(row["high"]), l=float(row["low"]),
                      o=float(row["open"]), c=float(row["close"]),
                      v=float(row["volume"]), t=row["date"],
                      index=v13)
            if v13 == 0:
                # 首次：只追加 src（process_cl_kline 下轮处理）
                self.src_klines.append(k)
                v13 += 1
                continue
            last = self.src_klines[-1]
            if last["t"] > k["t"]:
                continue                       # 乱序跳过
            if last["t"] == k["t"] and last["c"] == k["c"] and last["o"] == k["o"] \
                    and last["h"] == k["h"] and last["l"] == k["l"] and last["v"] == k["v"]:
                continue                       # 全等去重
            if last["t"] == k["t"]:
                last.update(h=k["h"], l=k["l"], o=k["o"], c=k["c"], v=k["v"])
            else:
                self.src_klines.append(k)
                v13 += 1
            self.process_cl_kline()
            self.process_idx()                       # macd 指标（每根都算）
            v19 = self.process_fx()                  # 分型
            v20 = self.process_bi() if v19 else False
            # 重算所有笔的 high/low（869 延伸/826 更新后端点已变）
            for _bi in self.bis:
                self._process_line_hl(_bi, "bi")
            if self.config.get("__cal_xd", 1):
                v21 = self.process_up_line("bi") if v20 else False
            else:
                v21 = False
            if self.config.get("__cal_zsd", 0):
                v22 = self.process_up_line("xd") if v21 else False
            else:
                v22 = False
            if self.config.get("__cal_qsd", 0):
                v23 = self.process_up_line("zsd") if v22 else False
            else:
                v23 = False
            if v20:
                self.process_zs(["bi"])
            if self.config.get("__cal_xd_zss", 1) and v21:
                self.process_zs(["xd"])
            if self.config.get("__cal_zsd_zss", 0) and v22:
                self.process_zs(["zsd"])
            if self.config.get("__cal_qsd_zss", 0) and v23:
                self.process_zs(["qsd"])
            if v20:
                self.process_mmd(["bi"])
            if self.config.get("__cal_xd_mmd", 1) and v21:
                self.process_mmd(["xd"])
            if self.config.get("__cal_zsd_mmd", 0) and v22:
                self.process_mmd(["zsd"])
            if self.config.get("__cal_qsd_mmd", 0) and v23:
                self.process_mmd(["qsd"])
        # 统一重算笔的 high/low（端点极值，对齐）
        for bi in self.bis:
            self._process_line_hl(bi, "bi")
        return self


# ==================== 线段对象 ====================
class TZXL:
    """特征序列元素"""
    __slots__ = ("bh_direction", "line", "pre_line", "max", "min", "lines",
                 "line_bad", "index", "is_up_line", "done")

    def __init__(self, bh_direction, line, pre_line, line_bad, done):
        self.bh_direction = bh_direction
        self.line = line
        self.pre_line = pre_line
        self.line_bad = line_bad
        self.is_up_line = False
        self.lines = [line]
        self.done = done
        self.max = 0
        self.min = 0
        self.index = line.index
        self.update_maxmin()

    def update_maxmin(self):
        if self.bh_direction == "up":
            self.max = max(l.high for l in self.lines)
            self.min = max(l.low for l in self.lines)
        else:
            self.max = min(l.high for l in self.lines)
            self.min = min(l.low for l in self.lines)

    def __str__(self):
        return f"done {self.done} max {self.max} min {self.min} line_bad {self.line_bad} " \
               f"line {self.line} pre_line {self.pre_line} num {len(self.lines)}"


class XLFX:
    """特征序列分型（cl_interface.py 定义）"""
    __slots__ = ("type", "xl", "xls", "qk", "bh_type",
                 "is_line_bad", "fx_high", "fx_low", "done")

    def __init__(self, _type, xl, xls, done=True):
        self.type = _type       # ding / di
        self.xl = xl            # 分型所在的特征序列元素（TZXL）
        self.xls = xls          # 特征序列元素列表（3个，可含 None）
        self.qk = False         # 分型是否有缺口
        self.is_line_bad = False  # 是否是一笔破坏分型
        self.fx_high = max(x.max for x in xls if x is not None)
        self.fx_low = min(x.min for x in xls if x is not None)
        self.done = done
        self.bh_type = None

    @property
    def high(self):
        return self.xl.max

    @property
    def low(self):
        return self.xl.min

    def get_last_xl(self):
        return [x for x in self.xls if x is not None][-1]

    def __str__(self):
        return f"XLFX type : {self.type} done : {self.done} qk : {self.qk} " \
               f"high : {self.high} low : {self.low} xl : {self.xl}"


class XD:
    """线段"""
    __slots__ = ("index", "start", "end", "start_line", "end_line", "type",
                 "ding_fx", "di_fx", "is_split", "tzxls", "done", "not_del",
                 "not_yx", "high", "low", "zs_high", "zs_low", "ld", "mmds",
                 "bcs", "zs_type_mmds", "zs_type_bcs", "default_zs_type")

    def __init__(self, start, end, start_line, end_line, _type,
                 ding_fx, di_fx, index, default_zs_type):
        self.index = index
        self.start = start
        self.end = end
        self.start_line = start_line
        self.end_line = end_line
        self.type = _type
        self.ding_fx = ding_fx
        self.di_fx = di_fx
        self.is_split = ""
        self.tzxls = []
        self.done = False
        self.not_del = False
        self.not_yx = False
        self.high = None
        self.low = None
        self.zs_high = None
        self.zs_low = None
        self.ld = None
        self.mmds = []
        self.bcs = []
        self.zs_type_mmds = {}
        self.zs_type_bcs = {}
        self.default_zs_type = default_zs_type

    def is_done(self):
        return self.done

    def is_qk(self):
        return self.ding_fx.qk if self.type == "up" else self.di_fx.qk

    def ding_high(self):
        return self.end.val if self.type == "up" else self.start.val

    def di_low(self):
        return self.end.val if self.type == "down" else self.start.val


# ==================== 线段方法 ====================
def _xd_line_to_tzxl(self, base_lines, bh_direction, _l):
    """线转特征序列元素（_xd_line_to_tzxl）"""
    return TZXL(bh_direction, _l, base_lines[_l.index - 1], False, True)


def _xd_get_up_line_next_line(self, base_lines, up_line, is_next_line=True):
    """获取 up_line 之后的下一条线（_xd_get_up_line_next_line）：
    is_next_line=True 时若首元素与线段同向则删除，False 时不删除"""
    if up_line.type == "up":
        if is_next_line:
            v211 = base_lines[up_line.ding_fx.xl.pre_line.index + 1:]
            if len(v211) == 0:
                return []
            if v211[0].type == "up":
                del v211[0]
            return v211
        else:
            v211 = base_lines[up_line.di_fx.xl.pre_line.index + 1:]
            if len(v211) == 0:
                return []
            return v211
    else:
        if is_next_line:
            v211 = base_lines[up_line.di_fx.xl.pre_line.index + 1:]
            if len(v211) == 0:
                return []
            if v211[0].type == "down":
                del v211[0]
            return v211
        else:
            v211 = base_lines[up_line.ding_fx.xl.pre_line.index + 1:]
            if len(v211) == 0:
                return []
            return v211


def _xd_done_pre_xd(self, up_lines):
    """将 up_lines 中未完成的线段全部标记完成"""
    for up_line in up_lines:
        if up_line.done is False:
            up_line.done = True
    return True


def _qk_num(self, klines):
    """K线序列内缺口数量（_qk_num）"""
    num = 0
    for i in range(len(klines) - 1):
        k1, k2 = klines[i], klines[i + 1]
        if isinstance(k1, dict):
            h1, l1, h2, l2 = k1["h"], k1["l"], k2["h"], k2["l"]
        else:
            h1, l1, h2, l2 = k1.h, k1.l, k2.h, k2.l
        if self._cross_qujian([h1, l1], [h2, l2]) is None:
            num += 1
    return num


def _xd_cal_line_xlfx(self, lines, fx_type="ding", bh_type="no_bh",
                      no_done_fx=False, three_fx=False):
    """计算线的序列分型（_xd_cal_line_xlfx）：
    返回 (xls 列表, fxs 列表)"""
    if len(lines) == 0:
        return [], []
    # 目标线类型：ding 分型看 down 线，di 分型看 up 线
    target = "up" if fx_type == "ding" else "down"
    xls = []
    for i in range(len(lines)):
        line = lines[i]
        if (fx_type == "ding" and line.type == "down") or \
           (fx_type == "di" and line.type == "up"):
            pre_line = lines[i - 1] if i > 0 else line
            xl = TZXL(target, line, pre_line, False, True)
            if len(xls) == 0:
                xls.append(xl)
                continue
            pre_xl = xls[-1]
            # 包含处理（更新 pre 对象并保留）
            if pre_xl.max >= xl.max and pre_xl.min <= xl.min:
                # pre 包含 cur：合并到 pre（保留 pre 对象）
                if target == "up":
                    if xl.max >= pre_xl.max:
                        pre_xl.line = xl.line
                        pre_xl.pre_line = xl.pre_line
                else:
                    if xl.min <= pre_xl.min:
                        pre_xl.line = xl.line
                        pre_xl.pre_line = xl.pre_line
                pre_xl.lines.append(line)
                pre_xl.line_bad = xl.line_bad
                pre_xl.update_maxmin()
            elif pre_xl.max <= xl.max and pre_xl.min >= xl.min:
                # cur 包含 pre
                if bh_type == "no_bh":
                    xl.line_bad = True
                    xls.append(xl)
                else:  # bh：合并到 pre（保留 pre 对象）
                    # 4143/4148：line 用 >=，pre_line 用 >
                    if target == "up":
                        if xl.max >= pre_xl.max:
                            pre_xl.line = xl.line
                        if xl.max > pre_xl.max:
                            pre_xl.pre_line = xl.pre_line
                    else:
                        # 4154/4159：line 用 <=，pre_line 用 <
                        if xl.min <= pre_xl.min:
                            pre_xl.line = xl.line
                        if xl.min < pre_xl.min:
                            pre_xl.pre_line = xl.pre_line
                    pre_xl.lines.append(line)
                    pre_xl.line_bad = xl.line_bad
                    pre_xl.update_maxmin()
            else:
                xls.append(xl)
    if len(xls) == 0:
        return [], []
    # 特征序列分型
    start_val = lines[0].start.val
    fxs = []
    v269 = []
    v270 = xls[0].max
    v271 = xls[0].min
    for i in range(1, len(xls)):
        pre_xl = xls[i - 1]
        now_xl = xls[i]
        v270 = max(v270, now_xl.max)
        v271 = min(v271, now_xl.min)
        if len(xls) > i + 1:
            next_xl = xls[i + 1]
        else:
            next_xl = None
            # 最后元素：no_done_fx 且 now_xl 是累积极值
            # （max==v270 / min==v271）时才生成 done=False 分型，否则跳过
            if not no_done_fx:
                continue
            if fx_type == "ding" and now_xl.max == v270:
                pass
            elif fx_type == "di" and now_xl.min == v271:
                pass
            else:
                continue
        three = [pre_xl, now_xl, next_xl]
        if fx_type == "ding":
            if pre_xl.max <= now_xl.max and (next_xl is None or now_xl.max >= next_xl.max):
                # 顶分型：前面所有元素 max 小于当前 min → 缺口
                v276 = max(x.max for x in xls[:i])
                v277 = (v276 < now_xl.min)
                xlfx = XLFX("ding", now_xl, three, next_xl is not None)
                xlfx.qk = v277
                xlfx.bh_type = bh_type
                # ：v280 = min(v279.lines 的 .low)；is_line_bad = v280 < start_val
                v279 = next_xl if next_xl is not None else now_xl
                xlfx.is_line_bad = (min(x.low for x in v279.lines) < start_val)
                fxs.append(xlfx)
                v269.append(xlfx)
        else:
            if pre_xl.min >= now_xl.min and (next_xl is None or now_xl.min <= next_xl.min):
                # 底分型：前面所有元素 min 大于当前 max → 缺口
                v281 = min(x.min for x in xls[:i])
                v277 = (v281 > now_xl.max)
                xlfx = XLFX("di", now_xl, three, next_xl is not None)
                xlfx.qk = v277
                xlfx.bh_type = bh_type
                # ：v282 = min(v279.lines 的 .low)；is_line_bad = v282 > start_val
                v279 = next_xl if next_xl is not None else now_xl
                xlfx.is_line_bad = (min(x.low for x in v279.lines) > start_val)
                fxs.append(xlfx)
                v269.append(xlfx)
    # 主循环无分型时：three_fx 的 3 根线单元素特例
    if len(v269) == 0:
        if three_fx and len(lines) == 3 and len(xls) == 1:
            # 行为 A：xls[0].pre_line 指向 lines[2]，强制生成 done=False 分型
            # （4249/4252：up 时 lines[2].high == 最高；down 时 lines[2].low == 最低）
            if lines[0].type == "up":
                cond = lines[2].high == max(l.high for l in lines)
            else:
                cond = lines[2].low == min(l.low for l in lines)
            if cond:
                xls[0].pre_line = lines[2]
                xlfx = XLFX(fx_type, xls[0], [xls[0]], False)
                xlfx.qk = False
                xlfx.bh_type = bh_type
                fxs.append(xlfx)
    # 行为 B：no_done_fx 且 lines>3 根时，最后元素方向与
    # 分型目标相反且是区间最高（ding）/最低（di）点时，生成其 done=False 分型
    # 注意：v283/v284 用 lines[1:]（排除第一根，实测）
    if no_done_fx and len(lines) > 3:
        v283 = max(l.high for l in lines[1:])
        v284 = min(l.low for l in lines[1:])
        if fx_type == "ding":
            if lines[-1].type == "up" and lines[-1].high == v283:
                v285 = TZXL("up", lines[-1], lines[-1], False, False)
                xlfx = XLFX("ding", v285, [v285], False)
                xlfx.qk = False
                xlfx.bh_type = bh_type
                fxs.append(xlfx)
        else:
            if lines[-1].type == "down" and lines[-1].low == v284:
                v285 = TZXL("down", lines[-1], lines[-1], False, False)
                xlfx = XLFX("di", v285, [v285], False)
                xlfx.qk = False
                xlfx.bh_type = bh_type
                fxs.append(xlfx)
    return xls, fxs


def _xd_get_next_tzfx(self, dings, dis, find_type):
    """找下一个特征分型（ _xd_get_next_tzfx 精确反编译）：
    ding：dings/dis 均降序；di：dis/dings 均降序"""
    if find_type == "ding":
        if len(dings) == 0:
            return None
        dings = sorted(dings, key=lambda x: x.high, reverse=True)
        if len(dis) == 0:
            return dings[0]
        dis = sorted(dis, key=lambda x: x.xl.pre_line.index, reverse=True)
        for i in range(len(dings)):
            v61 = dings[i]
            v221 = dings[i + 1] if i + 1 < len(dings) else None
            v222 = True
            for v62 in dis:
                if v62.xl.pre_line.index < v61.xl.pre_line.index:
                    if v61.xl.pre_line.index - v62.xl.pre_line.index >= 2:
                        if v61.high > v62.low:
                            if v221 is not None:
                                if v221.xl.pre_line.index < v62.xl.pre_line.index:
                                    if v221.high > v62.low:
                                        if v62.xl.pre_line.index - v221.xl.pre_line.index >= 2:
                                            v222 = False
                            else:
                                v222 = False
            if v222:
                return v61
        return dings[0]
    else:  # find_type == 'di'（ 对称）
        if len(dis) == 0:
            return None
        # 实测（1973/1973 调用）：dis 按 low 升序（返回 low 最小的 di）
        dis = sorted(dis, key=lambda x: x.low)
        if len(dings) == 0:
            return dis[0]
        dings = sorted(dings, key=lambda x: x.xl.pre_line.index, reverse=True)
        for i in range(len(dis)):
            v62 = dis[i]
            v223 = dis[i + 1] if i + 1 < len(dis) else None
            v222 = True
            for v61 in dings:
                if v61.xl.pre_line.index < v62.xl.pre_line.index:
                    if v62.xl.pre_line.index - v61.xl.pre_line.index >= 2:
                        if v61.high > v62.low:
                            if v223 is not None:
                                if v223.xl.pre_line.index < v61.xl.pre_line.index:
                                    # 3341：next_di.low < ding.high（不是 high > low）
                                    if v223.low < v61.high:
                                        if v61.xl.pre_line.index - v223.xl.pre_line.index >= 2:
                                            v222 = False
                            else:
                                v222 = False
            if v222:
                return v62
        return dis[0]


def _xd_get_up_line_tzxl_info(self, base_lines, up_lines, cal_type):
    """计算线段后的特征序列顶底分型（_xd_get_up_line_tzxl_info）"""
    if len(up_lines) == 0:
        v171 = base_lines
        v172 = base_lines
    else:
        v64 = up_lines[-1]
        v171 = self._xd_get_up_line_next_line(base_lines, v64)
        v172 = self._xd_get_up_line_next_line(base_lines, v64, False)
    result = {"next_base_lines": v171, "line_base_lines": v172}
    if "di" in cal_type:
        v82, v83 = self._xd_cal_line_xlfx(v171, "di", "no_bh", True, True)
    else:
        v82, v83 = [], []
    if "bh_di" in cal_type:
        v90, v91 = self._xd_cal_line_xlfx(v171, "di", "bh", False, False)
    else:
        v90, v91 = [], []
    if "ding" in cal_type:
        v84, v85 = self._xd_cal_line_xlfx(v171, "ding", "no_bh", True, True)
    else:
        v84, v85 = [], []
    if "bh_ding" in cal_type:
        v92, v93 = self._xd_cal_line_xlfx(v171, "ding", "bh", False, False)
    else:
        v92, v93 = [], []
    if "line_di" in cal_type:
        v86, v87 = self._xd_cal_line_xlfx(v172, "di", "no_bh", True, True)
    else:
        v86, v87 = [], []
    if "bh_line_di" in cal_type:
        v94, v95 = self._xd_cal_line_xlfx(v172, "di", "bh", False, False)
    else:
        v94, v95 = [], []
    if "line_ding" in cal_type:
        v88, v89 = self._xd_cal_line_xlfx(v172, "ding", "no_bh", True, True)
    else:
        v88, v89 = [], []
    if "bh_line_ding" in cal_type:
        v96, v97 = self._xd_cal_line_xlfx(v172, "ding", "bh", False, False)
    else:
        v96, v97 = [], []
    result.update({
        "di": {"xls": v82, "fxs": v83},
        "bh_di": {"xls": v90, "fxs": v91},
        "ding": {"xls": v84, "fxs": v85},
        "bh_ding": {"xls": v92, "fxs": v93},
        "line_di": {"xls": v86, "fxs": v87},
        "bh_line_di": {"xls": v94, "fxs": v95},
        "line_ding": {"xls": v88, "fxs": v89},
        "bh_line_ding": {"xls": v96, "fxs": v97},
    })
    return result


def _xd_check_xd_is_ok(self, base_lines, up_line, lines):
    """检查给定的线列表能否组成线段（_xd_check_xd_is_ok）"""
    if len(lines) in (1, 3, 5):
        return True, [], [], ""
    if len(lines) % 2 == 0:
        return False, [], [], ""
    v7 = self.create_dn_zs("bi", lines)
    v200 = []
    v201 = []
    if self.config.get("xd_allow_split_zs_more_line", "1"):
        v202 = max(self.config.get("xd_zs_max_lines_split", 11), 9)
        for zs in v7:
            if len(zs.lines) >= v202:
                v200.append(f"段内中枢线段超过{v202}")
                v201.extend(zs.lines)
    if self.config.get("xd_allow_split_zs_no_direction", "1"):
        for i in range(1, len(v7)):
            v203 = v7[i - 1]
            v204 = v7[i]
            # ：后中枢区间完全低于/高于前中枢（zg/zd 不重叠反向）→ 段内不同向中枢拆分
            if up_line.type == "up" and v204.zg < v203.zd:
                v200.append("段内不同向中枢拆分")
                v201.extend(v203.lines)
                v201.extend(v204.lines)
            elif up_line.type == "down" and v204.zd > v203.zg:
                v200.append("段内不同向中枢拆分")
                v201.extend(v203.lines)
                v201.extend(v204.lines)
    if self.config.get("xd_allow_split_zs_kz", "0"):
        for i in range(1, len(v7)):
            v203 = v7[i - 1]
            v204 = v7[i]
            if self.zss_is_qs(v203, v204) is not None:
                continue
            v200.append("中枢扩展")
            v201.extend(v203.lines)
            v201.extend(v204.lines)
    if self.config.get("xd_allow_split_no_highlow", "1"):
        if len(lines) >= 9:
            v205 = max(l.high for l in lines)
            v206 = min(l.low for l in lines)
            # ：线段的起始/结束分型是否为最高最低（非标准线段）
            if up_line.type == "up":
                cond = up_line.start.val != v206 or up_line.end.val != v205
            else:
                cond = up_line.start.val != v205 or up_line.end.val != v206
            if cond:
                v200.append("非标准线段（起始非最高最低）")
                v201.extend(lines)
    if len(v200) == 0:
        return True, [], [], ""
    v201 = sorted(set(v201), key=lambda l: l.index)
    v207 = "/".join(sorted(set(v200), reverse=True))
    v208 = "/".join(sorted(set(v200), reverse=False))
    # 返回：v178=up 线（high 降序），v179=down 线（low 升序），v180=原因
    return False, sorted([l for l in v201 if l.type == "up"],
                         key=lambda l: l.high, reverse=True), \
                  sorted([l for l in v201 if l.type == "down"],
                         key=lambda l: l.low), v208


def _xd_add_up_line(self, base_line_type, base_lines, up_lines, default_zs_type,
                    up_line_type, up_line, tzxlfx, tzxls, is_split,
                    not_del=False, not_yx=False):
    """新增一个线段（_xd_add_up_line）"""
    if tzxlfx is None:
        return False
    if up_lines[-1].index != up_line.index:
        raise Exception("up_lines 最后一个与 up_line 不一致")
    if up_line.end_line.index >= base_lines[-1].index:
        return False
    v173 = base_lines[up_line.end_line.index + 1]
    v174 = tzxlfx.xl.pre_line
    if v174.index < v173.index:
        return False
    v175 = XD(v173.start, v174.end, v173, v174,
              "up" if tzxlfx.type == "ding" else "down",
              tzxlfx if tzxlfx.type == "ding" else up_line.ding_fx,
              up_line.di_fx if tzxlfx.type == "ding" else tzxlfx,
              up_line.index + 1,
              default_zs_type)
    v175.is_split = is_split
    v175.tzxls = tzxls
    v175.done = (tzxlfx.done and tzxlfx.qk is False)
    v175.not_del = not_del
    v175.not_yx = not_yx
    self._process_line_hl(v175, up_line_type)
    if v175.end.index <= v175.start.index:
        raise Exception("线段 end 不大于 start")
    if v175.high < v175.low:
        return False
    up_lines.append(v175)
    self.process_zs([up_line_type])
    self.process_mmd([up_line_type])
    self._xd_check_and_split_up_line(base_line_type, base_lines, up_lines,
                                     default_zs_type, up_line_type, v175)
    return v175


def _xd_update_tzfx(self, base_line_type, base_lines, up_lines, up_line_type,
                    default_zs_type, up_line, tzxls, tzxlfx, is_split):
    """更新线段的特征序列对象（_xd_update_tzfx）"""
    v173 = up_line.start_line
    v174 = tzxlfx.xl.pre_line
    if tzxlfx.done is False:
        if tzxlfx.type == "ding":
            v176 = [l for l in base_lines[v174.index:]
                    if l.index >= v174.index and l.type == "up" and l.high >= v173.low]
            v176 = sorted(v176, key=lambda l: l.high, reverse=True)
        else:
            v176 = [l for l in base_lines[v174.index:]
                    if l.index >= v174.index and l.type == "down" and l.low <= v173.high]
            v176 = sorted(v176, key=lambda l: l.low)
        v174 = v176[0]
    if v174.index < v173.index:
        print(f"[_xd_update_tzfx] DEBUG: up_line={up_line.start_line.index}-{up_line.end_line.index} "
              f"type={up_line.type} done={up_line.done} not_yx={up_line.not_yx} "
              f"tzxlfx={tzxlfx.type} done={tzxlfx.done} qk={tzxlfx.qk} xl_pl={tzxlfx.xl.pre_line.index if tzxlfx.xl else None} "
              f"v173={v173.index} v174={v174.index} bis={len(base_lines)} xds={[(x.start_line.index, x.end_line.index, x.type) for x in up_lines]}")
        raise Exception("特征序列 pre_line 在 start_line 之前")
    if up_line.type == "up" and v174.high < v173.low:
        raise Exception("up 线段特征序列 high 小于 start_line.low")
    if up_line.type == "down" and v174.low > v173.high:
        raise Exception("down 线段特征序列 low 大于 start_line.high")
    up_line.end_line = v174
    up_line.end = v174.end
    if up_line.type == "up":
        up_line.ding_fx = tzxlfx
    else:
        up_line.di_fx = tzxlfx
    up_line.tzxls = tzxls
    up_line.done = (tzxlfx.done and tzxlfx.qk is False)
    up_line.is_split = is_split
    self._process_line_hl(up_line, up_line_type)
    self.process_zs([up_line_type])
    self.process_mmd([up_line_type])
    if up_line.start_line.type != up_line.end_line.type:
        raise Exception("start_line 与 end_line 类型不一致")
    self._xd_check_and_split_up_line(base_line_type, base_lines, up_lines,
                                     default_zs_type, up_line_type, up_line)
    return up_line


def _xd_split_optimal_hl_ll(self, base_lines, split_up_line, line_ups, line_downs):
    """线段查找最优分割线（_xd_split_optimal_hl_ll）"""
    if not (len(line_downs) >= 3 or len(line_ups) >= 3):
        return (None, None)
    v212 = base_lines[split_up_line.start_line.index: split_up_line.end_line.index + 1]
    v213 = max(l.high for l in v212)
    v214 = min(l.low for l in v212)
    if split_up_line.high != v213 or split_up_line.low != v214:
        if split_up_line.type == "up":
            for v182 in line_downs:
                for v181 in line_ups:
                    if v182.index > v181.index and v182.end.val < v181.end.val:
                        return (v181, v182)
        elif split_up_line.type == "down":
            for v181 in line_ups:
                for v182 in line_downs:
                    if v181.index > v182.index and v181.end.val > v182.end.val:
                        return (v181, v182)
        return (None, None)

    def diff_val(hl, ll):
        v217 = 0
        if split_up_line.type == "up":
            v218 = base_lines[split_up_line.start_line.index: hl.index + 1]
            v219 = base_lines[hl.index + 1: ll.index + 1]
            v220 = base_lines[ll.index + 1: split_up_line.end_line.index + 1]
            v217 += v218[0].low - min(l.low for l in v218)
            v217 += max(l.high for l in v218) - v218[-1].high
            v217 += max(l.high for l in v219) - v219[0].high
            v217 += v219[-1].low - min(l.low for l in v219)
            v217 += v220[0].low - min(l.low for l in v220)
            v217 += max(l.high for l in v220) - v220[-1].high
        else:
            v218 = base_lines[split_up_line.start_line.index: ll.index + 1]
            v219 = base_lines[ll.index + 1: hl.index + 1]
            v220 = base_lines[hl.index + 1: split_up_line.end_line.index + 1]
            v217 += max(l.high for l in v218) - v218[0].high
            v217 += v218[-1].low - min(l.low for l in v218)
            v217 += v219[0].low - min(l.low for l in v219)
            v217 += max(l.high for l in v219) - v219[-1].high
            v217 += max(l.high for l in v220) - v220[0].high
            v217 += v220[-1].low - min(l.low for l in v220)
        return v217

    v215 = []
    if split_up_line.type == "up":
        for v182 in line_downs:
            for v181 in line_ups:
                if v182.index - v181.index >= 2 \
                   and v181.index - split_up_line.start_line.index >= 2 \
                   and split_up_line.end_line.index - v182.index >= 2 \
                   and v182.end.val < v181.end.val:
                    v215.append({"hl": v181, "ll": v182,
                                 "diff_val": diff_val(v181, v182)})
    elif split_up_line.type == "down":
        for v181 in line_ups:
            for v182 in line_downs:
                if v181.index - v182.index >= 2 \
                   and v182.index - split_up_line.start_line.index >= 2 \
                   and split_up_line.end_line.index - v181.index >= 2 \
                   and v181.end.val > v182.end.val:
                    v215.append({"hl": v181, "ll": v182,
                                 "diff_val": diff_val(v181, v182)})
    if len(v215) == 0:
        return (None, None)
    v215 = sorted(v215, key=lambda x: x["diff_val"])
    return (v215[0]["hl"], v215[0]["ll"])


def _xd_check_and_split_up_line(self, base_line_type, base_lines, up_lines,
                                default_zs_type, up_line_type, up_line):
    """检查并分割线段（_xd_check_and_split_up_line）"""
    if len(up_lines) <= 1:
        return False
    v177, v178, v179, v180 = self._xd_check_xd_is_ok(
        base_lines, up_line,
        base_lines[up_line.start_line.index: up_line.end_line.index + 1])
    if v177:
        return False
    v181, v182 = self._xd_split_optimal_hl_ll(base_lines, up_line, v178, v179)
    if v181 is None or v182 is None:
        return False
    del up_lines[-1]
    up_lines[-1].not_del = True
    up_lines[-1].not_yx = True
    self._xd_done_pre_xd(up_lines)
    if up_line.type == "up":
        v183 = self._xd_line_to_tzxl(base_lines, "up", base_lines[v181.index - 1])
        v184 = self._xd_line_to_tzxl(base_lines, "up", base_lines[v181.index + 1])
        v185 = XLFX("ding", v184, [v183, v184, None], True)
        v185.bh_type = up_line.ding_fx.bh_type
        v186 = self._xd_add_up_line(base_line_type, base_lines, up_lines,
                                    default_zs_type, up_line_type,
                                    up_lines[-1], v185, [v183, v184, None],
                                    v180, False, False)
        if v186 is False:
            return True
        if up_lines[-1].type != "up":
            return True
        v187 = base_lines[up_lines[-1].end_line.index + 1:]
        v188, v189 = self._xd_cal_line_xlfx(v187, "di", "no_bh", True, True)
        if len(v189) > 0 and v189[0].xl.pre_line.low <= v182.low:
            v190 = v189[0]
            v191 = v188
        else:
            v192 = self._xd_line_to_tzxl(base_lines, "down", base_lines[v182.index - 1])
            v193 = self._xd_line_to_tzxl(base_lines, "down", base_lines[v182.index + 1])
            v191 = [v192, v193, None]
            v190 = XLFX("di", v193, v191, False)
            v190.bh_type = up_line.ding_fx.bh_type
        v194 = base_lines[v190.xl.pre_line.index + 1:]
        v195, v196 = self._xd_cal_line_xlfx(v194, "ding", "no_bh", True, True)
        v197 = False
        if len(v196) > 0 and v196[0].xl.pre_line.high == min(l.high for l in v194):
            v190.done = True
            v197 = True
        self._xd_add_up_line(base_line_type, base_lines, up_lines,
                             default_zs_type, up_line_type,
                             up_lines[-1], v190, v191, v180, False, v197)
    elif up_line.type == "down":
        v192 = self._xd_line_to_tzxl(base_lines, "down", base_lines[v182.index - 1])
        v193 = self._xd_line_to_tzxl(base_lines, "down", base_lines[v182.index + 1])
        v190 = XLFX("di", v193, [v192, v193, None], True)
        v190.bh_type = up_line.di_fx.bh_type
        v186 = self._xd_add_up_line(base_line_type, base_lines, up_lines,
                                    default_zs_type, up_line_type,
                                    up_lines[-1], v190, [v192, v193, None],
                                    v180, False, False)
        if v186 is False:
            return True
        if up_lines[-1].type != "down":
            return True
        v187 = base_lines[up_lines[-1].end_line.index + 1:]
        v188, v189 = self._xd_cal_line_xlfx(v187, "ding", "no_bh", True, True)
        if len(v189) > 0 and v189[0].xl.pre_line.high >= v181.high:
            v185 = v189[0]
            v199 = v188
        else:
            v183 = self._xd_line_to_tzxl(base_lines, "up", base_lines[v181.index - 1])
            v184 = self._xd_line_to_tzxl(base_lines, "up", base_lines[v181.index + 1])
            v199 = [v183, v184, None]
            v185 = XLFX("ding", v184, v199, False)
            v185.bh_type = up_line.di_fx.bh_type
        v194 = base_lines[v185.xl.pre_line.index + 1:]
        v195, v196 = self._xd_cal_line_xlfx(v194, "di", "no_bh", True, True)
        v197 = False
        if len(v196) > 0 and v196[0].xl.pre_line.low == min(l.low for l in v194):
            v190.done = True
            v197 = True
        self._xd_add_up_line(base_line_type, base_lines, up_lines,
                             default_zs_type, up_line_type,
                             up_lines[-1], v185, v199, v180, not_yx=v197)
    return True


def default_xd_zs_type(self):
    return self.config.get("zs_xd_type", ["zs_type_bz"])[0]


def default_bi_zs_type(self):
    return self.config.get("zs_bi_type", ["zs_type_bz"])[0]


def _line_ukey(self, _l):
    """线唯一键（_line_ukey）"""
    ukey = f"{_l.index}_{_l.start.k.date}_{_l.end.k.date}_{_l.start.val}_{_l.end.val}_" \
           f"{_l.start.index}_{_l.end.index}_{_l.start.done}_{_l.end.done}_{_l.is_done()}"
    if isinstance(_l, XD):
        ukey += f"__start_line__{self._line_ukey(_l.start_line)}"
        ukey += f"__end_line__{self._line_ukey(_l.end_line)}"
        ukey += f"__xl_ding__{self._line_ukey(_l.ding_fx.xl.pre_line)}"
        ukey += f"__xl_di__{self._line_ukey(_l.di_fx.xl.pre_line)}"
    return ukey


def _line_qj(self, line, qj_type):
    """线区间（_line_qj）"""
    if qj_type in ("bi_qj_dd", "xd_qj_dd", "zsd_qj_dd"):
        return [line.ding_high(), line.di_low()]
    if qj_type in ("bi_qj_ck", "xd_qj_ck", "zsd_qj_ck"):
        return [max(line.start.high(), line.end.high()),
                min(line.start.low(), line.end.low())]
    if qj_type in ("bi_qj_k", "xd_qj_k", "zsd_qj_k"):
        return [max(line.start.k.h, line.end.k.h),
                min(line.start.k.l, line.end.k.l)]
    raise Exception(f"区间类型错误 {qj_type}")


def _line_zs_hl(self, _l):
    """中枢用线区间（_line_zs_hl）"""
    if self.config.get("zs_qj", "zs_qj_dd") == "zs_qj_dd":
        return [_l.ding_high(), _l.di_low()]
    if self.config.get("zs_qj") == "zs_qj_ck":
        return [max(_l.start.high(), _l.end.high()),
                min(_l.start.low(), _l.end.low())]
    return [max(_l.start.k.h, _l.end.k.h), min(_l.start.k.l, _l.end.k.l)]


def _process_line_hl(self, line, line_type="bi"):
    """处理并计算线（笔、线段）实际高低点（_process_line_hl）"""
    if line_type == "bi":
        qj = self._line_qj(line, self.config.get("bi_qj", "bi_qj_dd"))
    elif line_type == "xd":
        qj = self._line_qj(line, self.config.get("xd_qj", "xd_qj_dd"))
    elif line_type in ("zsd", "qsd"):
        qj = self._line_qj(line, self.config.get("zsd_qj", "zsd_qj_dd"))
    else:
        raise Exception(f"line_type 错误 {line_type}")
    line.high = qj[0]
    line.low = qj[1]
    if isinstance(line, XD):
        line.start = line.start_line.start
        line.end = line.end_line.end
    zs_hl = self._line_zs_hl(line)
    line.zs_high = zs_hl[0]
    line.zs_low = zs_hl[1]
    return True


def process_up_line(self, base_line_type):
    """线段增量计算主流程（ process_up_line）"""
    if base_line_type == "bi":
        v54, v55, v56 = self.xds, "xd", self.bis
        v57 = self.default_xd_zs_type()
    elif base_line_type == "xd":
        v54, v55, v56 = self.zsds, "zsd", self.xds
        v57 = "zs_type_bz"
    elif base_line_type == "zsd":
        v54, v55, v56 = self.qsds, "qsd", self.zsds
        v57 = "zs_type_bz"
    else:
        raise Exception(f"base_line_type 错误 {base_line_type}")
    if len(v56) < 3:
        return False
    if len(v54) == 0:
        # 首线段：从特征序列分型创建
        v58 = self._xd_get_up_line_tzxl_info(v56, [], cal_type=["di", "ding"])
        v59 = sorted(v58["di"]["fxs"], key=lambda f: f.low)
        v60 = sorted(v58["ding"]["fxs"], key=lambda f: f.high,
                     reverse=True)
        for v61 in v60:
            for v62 in v59:
                if v61.high >= v62.low:
                    if abs(v61.xl.pre_line.index - v62.xl.pre_line.index) >= 2:
                        if v61.xl.pre_line.index > v62.xl.pre_line.index:
                            v63 = XD(v62.xl.line.start, v61.xl.pre_line.end,
                                     v62.xl.line, v61.xl.pre_line, "up",
                                     v61, v62, 0, v57)
                            v63.done = (v61.done is True)
                            v63.tzxls = v58["ding"]["xls"]
                            v54.append(v63)
                            return True
                        else:
                            v63 = XD(v61.xl.line.start, v62.xl.pre_line.end,
                                     v61.xl.line, v62.xl.pre_line, "down",
                                     v61, v62, 0, v57)
                            v63.done = (v62.done is True)
                            v63.tzxls = v58["di"]["xls"]
                            v54.append(v63)
                            return True
    if len(v54) == 0:
        return False
    self._process_line_hl(v54[-1], v55)
    v64 = v54[-1]
    # __cal_xd_youhua 优化检查
    if self.config.get("__cal_xd_youhua", "1"):
        if v64.end_line.index <= v56[-1].index and \
           self._line_ukey(v64.end_line) == self._line_ukey(v56[v64.end_line.index]):
            v65 = self._xd_get_up_line_tzxl_info(
                v56, v54, cal_type=["di", "ding", "line_di", "line_ding"])
            v66 = False
            if len(v65["di"]["fxs"]) != 0 or len(v65["ding"]["fxs"]) != 0:
                v66 = True
            if len(v65["line_ding"]["fxs"]) > 0:
                if v64.type == "up" and str(v64.ding_fx) != str(v65["line_ding"]["fxs"][-1]):
                    v66 = True
            if len(v65["line_di"]["fxs"]) > 0:
                if v64.type == "down" and str(v64.di_fx) != str(v65["line_di"]["fxs"][-1]):
                    v66 = True
            if v66 is False:
                return False
    # 删除失效的最后线段
    while True:
        if len(v54) == 0:
            break
        v64 = v54[-1]
        if v64.not_del:
            break
        if v64.is_done() is False and v64.is_qk() is False:
            del v54[-1]
            continue
        if v64.is_split != "":
            del v54[-1]
            continue
        if v64.end_line.index > v56[-1].index:
            del v54[-1]
            continue
        if self._line_ukey(v64.end_line) != self._line_ukey(v56[v64.end_line.index]):
            del v54[-1]
            continue
        if v64.ding_fx.get_last_xl().line.index > v56[-1].index:
            del v54[-1]
            continue
        if v64.di_fx.get_last_xl().line.index > v56[-1].index:
            del v54[-1]
            continue
        if v64.start.val != v56[v64.start_line.index].start.val:
            del v54[-1]
            continue
        if v64.end.val != v56[v64.end_line.index].end.val:
            del v54[-1]
            continue
        break
    if len(v54) == 0:
        return True
    # 主循环
    v46 = 0
    v67 = None
    while True:
        v46 += 1
        if v46 > 50:
            raise Exception(f"{self.code}-{self.frequency} {base_line_type}，"
                            f"时间：{self.src_klines[-1]['t']}")
        if len(v54) == 0:
            break
        if len(v54) >= 2:
            if v54[-1].is_done():
                self._xd_done_pre_xd(v54)
            if v54[-2].is_split != "" and v54[-2].is_done() is False:
                v54[-2].done = True
        v64 = v54[-1]
        v64.start = v64.start_line.start
        v64.end = v64.end_line.end
        v69 = self._line_ukey(v54[-1])
        # 未完成线段缺口确认
        for v70 in v54[::-1]:
            if v70.is_done():
                continue
            if v70.type == "up" and v70.is_qk() and v70.is_done() is False:
                _, v71 = self._xd_cal_line_xlfx(
                    self._xd_get_up_line_next_line(v56, v70), "di", "bh", False, False)
                if len(v71) > 0 and len(v71) > 0:
                    v70.done = True
            elif v70.type == "down" and v70.is_qk() and v70.is_done() is False:
                _, v72 = self._xd_cal_line_xlfx(
                    self._xd_get_up_line_next_line(v56, v70), "ding", "bh", False, False)
                if len(v72) > 0 and len(v72) > 0:
                    v70.done = True
        if v69 != self._line_ukey(v54[-1]):
            continue
        v64 = v54[-1]
        if v64.end_line.index > v56[-1].index or \
           self._line_ukey(v64.end_line) != self._line_ukey(v56[v64.end_line.index]):
            del v54[-1]
            continue
        # 笔破坏检查
        if self.config.get("xd_allow_bi_pohuai", "yes") != "no":
            v73 = True
            if v64.end_line.index - v64.start_line.index <= 2 and len(v54) >= 2:
                if v54[-2].is_qk() and v54[-1].is_done() is False:
                    v73 = False
            v74 = v56[v64.end_line.index + 1:]
            if v73:
                if v56[-1].index >= v64.end_line.index + 1:
                    if v64.type == "up" and v74[0].end.val < v64.start.val:
                        v75 = self._qk_num(
                            self.src_klines[v74[0].start.k.k_index:
                                            v74[0].end.k.k_index + 1]) > 0
                        if self.config.get("xd_allow_bi_pohuai") == "yes" or \
                           (self.config.get("xd_allow_bi_pohuai") == "yes_qk" and v75):
                            self._xd_done_pre_xd(v54)
                            v58 = self._xd_get_up_line_tzxl_info(
                                v56, v54, cal_type=["di", "ding"])
                            v76 = self._xd_get_next_tzfx(
                                v58["ding"]["fxs"], v58["di"]["fxs"], "di")
                            if v76 is not None and v76.low <= v74[0].low:
                                self._xd_add_up_line(
                                    base_line_type, v56, v54, v57, v55,
                                    v54[-1], v76, v58["di"]["xls"], "笔破坏")
                            else:
                                v77 = TZXL("down", v74[0], v74[0], True, False)
                                v78 = XLFX("di", v77, [v77, v77, None], False)
                                v78.is_line_bad = True
                                v79 = self._xd_add_up_line(
                                    base_line_type, v56, v54, v57, v55,
                                    v54[-1], v78, [v77], "笔破坏")
                                if v79 is not False and v79.index == v54[-1].index:
                                    v58 = self._xd_get_up_line_tzxl_info(
                                        v56, v54, cal_type=["di", "ding"])
                                    if len(v58["ding"]["fxs"]) > 0 and \
                                       v58["ding"]["fxs"][0].done:
                                        v54[-1].done = True
                                        v54[-1].not_del = True
                                        v54[-1].not_yx = True
                                        v80 = self._xd_get_next_tzfx(
                                            v58["ding"]["fxs"], v58["di"]["fxs"], "ding")
                                        if v80 is not None:
                                            self._xd_add_up_line(
                                                base_line_type, v56, v54, v57, v55,
                                                v54[-1], v80, v58["ding"]["xls"],
                                                "笔破坏")
                    elif v64.type == "down" and v74[0].end.val > v64.start.val:
                        v75 = self._qk_num(
                            self.src_klines[v74[0].start.k.k_index:
                                            v74[0].end.k.k_index + 1]) > 0
                        if self.config.get("xd_allow_bi_pohuai") == "yes" or \
                           (self.config.get("xd_allow_bi_pohuai") == "yes_qk" and v75):
                            self._xd_done_pre_xd(v54)
                            v58 = self._xd_get_up_line_tzxl_info(
                                v56, v54, cal_type=["di", "ding"])
                            v80 = self._xd_get_next_tzfx(
                                v58["ding"]["fxs"], v58["di"]["fxs"], "ding")
                            if v80 is not None and v80.high >= v74[0].high:
                                self._xd_add_up_line(
                                    base_line_type, v56, v54, v57, v55,
                                    v54[-1], v80, v58["ding"]["xls"], "笔破坏")
                            else:
                                v77 = TZXL("up", v74[0], v74[0], True, False)
                                v81 = XLFX("ding", v77, [v77, v77, None], False)
                                v81.is_line_bad = True
                                v79 = self._xd_add_up_line(
                                    base_line_type, v56, v54, v57, v55,
                                    v54[-1], v81, [v77], "笔破坏")
                                if v79 is not False and v79.index == v54[-1].index:
                                    v58 = self._xd_get_up_line_tzxl_info(
                                        v56, v54, cal_type=["di", "ding"])
                                    if len(v58["di"]["fxs"]) > 0 and \
                                       v58["di"]["fxs"][0].done:
                                        v54[-1].done = True
                                        v54[-1].not_del = True
                                        v54[-1].not_yx = True
                                        v76 = self._xd_get_next_tzfx(
                                            v58["ding"]["fxs"], v58["di"]["fxs"], "di")
                                        if v76 is not None:
                                            self._xd_add_up_line(
                                                base_line_type, v56, v54, v57, v55,
                                                v54[-1], v76, v58["di"]["xls"],
                                                "笔破坏")
        if v69 != self._line_ukey(v54[-1]):
            continue
        # 更新特征分型
        v65 = self._xd_get_up_line_tzxl_info(
            v56, v54, cal_type=["di", "ding", "line_di", "line_ding",
                                "bh_di", "bh_ding", "bh_line_di", "bh_line_ding"])
        v82, v83 = v65["di"]["xls"], v65["di"]["fxs"]
        v84, v85 = v65["ding"]["xls"], v65["ding"]["fxs"]
        v86, v87 = v65["line_di"]["xls"], v65["line_di"]["fxs"]
        v88, v89 = v65["line_ding"]["xls"], v65["line_ding"]["fxs"]
        v90, v91 = v65["bh_di"]["xls"], v65["bh_di"]["fxs"]
        v92, v93 = v65["bh_ding"]["xls"], v65["bh_ding"]["fxs"]
        v94, v95 = v65["bh_line_di"]["xls"], v65["bh_line_di"]["fxs"]
        v96, v97 = v65["bh_line_ding"]["xls"], v65["bh_line_ding"]["fxs"]
        # up 未完成线段的 ding_fx 更新
        if v64.type == "up" and v64.done is False:
            for v61 in v89:
                if v61.xl.pre_line.index == v64.ding_fx.xl.pre_line.index:
                    v64.ding_fx = v61
                    v64.done = (v61.qk is False and v61.done)
        if v64.type == "down" and v64.done is False:
            for v62 in v87:
                if v62.xl.pre_line.index == v64.di_fx.xl.pre_line.index:
                    v64.di_fx = v62
                    v64.done = (v62.qk is False and v62.done)
        if v69 != self._line_ukey(v54[-1]):
            continue
        # _xd_update_tzfx（特征分型延伸）
        v98 = self._xd_get_next_tzfx(v89, v83, "ding") if len(v89) > 0 else None
        if v64.type == "up" and v64.not_yx is False:
            if v98 is not None and \
               v98.xl.pre_line.end.val > v64.ding_fx.xl.pre_line.end.val and \
               v98.xl.pre_line.end.k.date > v64.ding_fx.xl.pre_line.end.k.date:
                self._xd_update_tzfx(base_line_type, v56, v54, v55, v57,
                                     v64, v88, v98, v64.is_split)
        v99 = self._xd_get_next_tzfx(v85, v87, "di") if len(v87) > 0 else None
        if v64.type == "down" and v64.not_yx is False:
            if v99 is not None and \
               v99.xl.pre_line.end.val < v64.di_fx.xl.pre_line.end.val and \
               v99.xl.pre_line.end.k.date > v64.di_fx.xl.pre_line.end.k.date:
                self._xd_update_tzfx(base_line_type, v56, v54, v55, v57,
                                     v64, v86, v99, v64.is_split)
        if v69 != self._line_ukey(v54[-1]):
            continue
        # _xd_add_up_line（新线段确认）
        if v64.type == "up" and len(v83) > 0:
            v76 = self._xd_get_next_tzfx(v93, v83, "di")
            self._xd_add_up_line(base_line_type, v56, v54, v57, v55,
                                 v54[-1], v76, v82, "")
        elif v64.type == "down" and len(v85) > 0:
            v80 = self._xd_get_next_tzfx(v85, v91, "ding")
            self._xd_add_up_line(base_line_type, v56, v54, v57, v55,
                                 v54[-1], v80, v84, "")
        # 收尾：ukey 无变化则退出
        v64 = v54[-1]
        v100 = self._line_ukey(v64)
        if v100 == v67:
            break
        v67 = v100
    if len(v54) == 0:
        return True
    ukey = self._line_ukey(v54[-1])
    if ukey != self._CL__last_up_line_end_ukey.get(base_line_type):
        self._CL__last_up_line_end_ukey[base_line_type] = ukey
        return True
    return False


# ==================== 方法绑定 ====================
_CL_METHODS = [
    "_xd_line_to_tzxl", "_xd_get_up_line_next_line", "_xd_done_pre_xd",
    "_qk_num", "_xd_cal_line_xlfx", "_xd_get_next_tzfx",
    "_xd_get_up_line_tzxl_info", "_xd_check_xd_is_ok", "_xd_add_up_line",
    "_xd_update_tzfx", "_xd_split_optimal_hl_ll", "_xd_check_and_split_up_line",
    "process_up_line", "default_xd_zs_type", "default_bi_zs_type", "_line_ukey", "_line_qj",
    "_line_zs_hl", "_process_line_hl",
]
for _name in _CL_METHODS:
    setattr(CL, _name, globals()[_name])


def _add_time(self, *args, **kwargs):
    pass


def _process_bz_zs(self, lines, zss, cal_line_type):
    """创建标准中枢（ _process_bz_zs：扩展 + 循环前缀创建）"""
    if len(zss) == 0:
        v249 = lines[-4:]
        v9 = self.create_zs(cal_line_type, None, v249)
        if v9:
            zss.append(v9)
        return True
    v10 = zss[-1]
    v118 = lines[v10.lines[0].index:]
    v250 = self.create_zs(cal_line_type, v10, v118)
    if v250 is None:
        del zss[-1]
        if len(zss) == 0:
            return True
        v10 = zss[-1]
    v211 = lines[v10.lines[-1].index:]
    if len(v211) < 4:
        return True
    # 循环后缀：v211[v5:] 逐个尝试创建新中枢，首个成功即跳出
    for v5 in range(len(v211)):
        v118 = v211[v5:]
        v120 = self.create_zs(cal_line_type, None, v118)
        if v120:
            v120.index = zss[-1].index + 1
            zss[-1].done = True
            zss.append(v120)
            break
    return True


def process_zs(self, cal_line_types):
    """计算中枢（ process_zs）"""
    if cal_line_types is None:
        return False
    if not isinstance(cal_line_types, (list, tuple)):
        cal_line_types = [cal_line_types]
    for v103 in cal_line_types:
        if v103 == "bi":
            if len(self.bis) < 4:
                continue
            zss = self.type_bi_zss.get("zs_type_bz", [])
            self._process_bz_zs(self.bis, zss, "bi")
            self.type_bi_zss["zs_type_bz"] = zss
            # 1670 行：bi_zss = type_bi_zss[default_bi_zs_type()]
            self.bi_zss = self.type_bi_zss.get(self.default_bi_zs_type(), [])
        elif v103 == "xd":
            if len(self.xds) < 4:
                continue
            zss = self.type_xd_zss.get("zs_type_bz", [])
            self._process_bz_zs(self.xds, zss, "xd")
            self.type_xd_zss["zs_type_bz"] = zss
            # 1724 行：xd_zss = type_xd_zss[default_xd_zs_type()]
            self.xd_zss = self.type_xd_zss.get(self.default_xd_zs_type(), [])
    return True


def process_mmd(self, *args, **kwargs):
    pass


def _line_get_zs_hl(self, _l):
    """线中枢区间（9.0.3 _line_get_zs_hl）"""
    return [_l.zs_high, _l.zs_low]


class ZS:
    """中枢对象（cl_interface.py 定义）"""
    __slots__ = ("zs_type", "start", "lines", "end", "zg", "zd", "gg", "dd",
                 "type", "index", "line_num", "level", "done", "real")

    def __init__(self, zs_type, start, end=None, zg=None, zd=None, gg=None,
                 dd=None, _type=None, index=0, line_num=0, level=0):
        self.zs_type = zs_type
        self.start = start
        self.lines = []
        self.end = end
        self.zg = zg
        self.zd = zd
        self.gg = gg
        self.dd = dd
        self.type = _type
        self.index = index
        self.line_num = line_num
        self.level = level
        self.done = False
        self.real = True


def create_zs(self, zs_type, zs, lines, max_line_num=999):
    """创建中枢（ create_zs 完整反编译）"""
    if len(lines) <= 3:
        return None
    v104 = []
    v105 = False
    v106 = self._cross_qujian(self._line_get_zs_hl(lines[1]),
                              self._line_get_zs_hl(lines[2]))
    if v106 is None:
        return None
    v106 = self._cross_qujian([v106["max"], v106["min"]],
                              self._line_get_zs_hl(lines[3]))
    if v106 is None:
        return None
    for v107 in lines:
        v108 = self._line_get_zs_hl(v107)
        v109 = self._cross_qujian([v106["max"], v106["min"]], v108)
        if v109:
            v104.append(v107)
            if self.config.get("zs_cd", "zs_cd_three") == "zs_cd_more":
                v106 = v109
            if len(v104) >= max_line_num:
                v105 = True
                break
        else:
            v105 = True
            break
    if len(v104) < 4:
        return None
    v110 = max(self._line_get_zs_hl(l)[0] for l in v104)
    v111 = min(self._line_get_zs_hl(l)[1] for l in v104)
    v112 = v104[-1]
    v113 = self._line_get_zs_hl(v112)
    v114 = False
    if v113[0] == v110 or v113[1] == v111:
        v114 = True
    if zs is None:
        zs = ZS(zs_type, v104[1].start, _type="zd")
    zs.start = v104[1].start
    zs.done = v105
    zs.lines = v104
    zs.zg = v106["max"]
    zs.zd = v106["min"]
    zs.line_num = len(zs.lines)
    zs.level = round(max(1, zs.line_num / 9), 2)
    if v105 or v114:
        if len(zs.lines) == 4:
            zs.end = v104[-1].end
        else:
            zs.end = v104[-2].end
        zs.gg = max(self._line_get_zs_hl(l)[0] for l in v104[1:-1])
        zs.dd = min(self._line_get_zs_hl(l)[1] for l in v104[1:-1])
    else:
        zs.end = v104[-1].end
        zs.gg = max(self._line_get_zs_hl(l)[0] for l in v104[1:])
        zs.dd = min(self._line_get_zs_hl(l)[1] for l in v104[1:])
    # 中枢方向：首尾线同向且满足位置关系才设方向，否则 zd
    if zs.lines[0].type == zs.lines[-1].type:
        v115 = self._line_get_zs_hl(zs.lines[0])
        v116 = self._line_get_zs_hl(zs.lines[-1])
        if zs.lines[0].type == "up":
            if v115[1] <= zs.dd and v116[0] >= zs.gg:
                zs.type = zs.lines[0].type
            else:
                zs.type = "zd"
        elif zs.lines[0].type == "down":
            if v115[0] >= zs.gg and v116[1] <= zs.dd:
                zs.type = zs.lines[0].type
            else:
                zs.type = "zd"
    else:
        zs.type = "zd"
    return zs


def create_dn_zs(self, line_type, lines, max_line_num=999, zs_include_last_line=True):
    """创建段内中枢（ create_dn_zs：lines[i:] 后缀循环）"""
    v7 = []
    i = 0
    while i < len(lines):
        v118 = lines[i:]
        if len(v118) == 0:
            break
        v119 = self.create_zs(line_type, None, v118, max_line_num=max_line_num)
        if v119 is None:
            i += 1
            continue
        v7.append(v119)
        i += max(len(v119.lines) - 1, 1)
    return v7

def zss_is_qs(self, one_zs, two_zs):
    """中枢趋势判断（ zss_is_qs 完整反编译）"""
    if one_zs.lines[-1].type != two_zs.lines[-1].type:
        return None
    if self.config.get("judge_zs_qs_level"):
        if one_zs.level != two_zs.level:
            return None
    if one_zs.zg < two_zs.zd:
        v147 = "up"
    elif one_zs.zd > two_zs.zg:
        v147 = "down"
    else:
        return None
    wzgx = self.config.get("zs_wzgx", "zs_wzgx_gd")
    if wzgx == "zs_wzgx_zgd":
        if v147 == "up" and two_zs.zd <= one_zs.zg:
            return None
        if v147 == "down" and two_zs.zg >= one_zs.zd:
            return None
    elif wzgx == "zs_wzgx_zggdd":
        if v147 == "up" and two_zs.zd <= one_zs.gg:
            return None
        if v147 == "down" and two_zs.zg >= one_zs.dd:
            return None
    elif wzgx == "zs_wzgx_gd":
        if v147 == "up" and two_zs.dd <= one_zs.gg:
            return None
        if v147 == "down" and two_zs.gg >= one_zs.dd:
            return None
    else:
        raise Exception("中枢类型配置错误")
    return v147


def create_zs_list(self, lines):
    """简化中枢创建（返回列表）"""
    return []


# ==================== LINE 通用方法（BI/XD 共享，cl_interface.py） ====================

def _line_get_ld(self, cl):
    """线的力度（get_ld：{'macd': query_macd_ld(cl, start, end)}）"""
    return {"macd": query_macd_ld(cl, self.start, self.end)}


def _line_get_mmds(self, zs_type=None):
    """返回当前线买卖点（过滤 zs.real）"""
    if zs_type is None:
        return [_m for _m in self.mmds if _m.zs.real]
    if zs_type not in self.zs_type_mmds.keys():
        return []
    return [_m for _m in self.zs_type_mmds[zs_type] if _m.zs.real]


def _line_add_mmd(self, name, zs, zs_type, msg=""):
    """添加买卖点"""
    mmd_obj = MMD(name, zs)
    mmd_obj.msg = msg
    if zs_type == self.default_zs_type:
        self.mmds.append(mmd_obj)
    if zs_type not in self.zs_type_mmds.keys():
        self.zs_type_mmds[zs_type] = []
    self.zs_type_mmds[zs_type].append(mmd_obj)
    return True


def _line_add_bc(self, _type, zs, compare_line, compare_lines, bc, zs_type):
    """添加背驰点"""
    bc_obj = BC(_type, zs, compare_line, compare_lines, bc)
    if zs_type == self.default_zs_type:
        self.bcs.append(bc_obj)
    if zs_type not in self.zs_type_bcs.keys():
        self.zs_type_bcs[zs_type] = []
    self.zs_type_bcs[zs_type].append(bc_obj)
    return True


def _line_line_mmds(self, zs_type=None):
    """返回当前线所有买卖点名称"""
    if zs_type is None:
        return [m.name for m in self.mmds if m.zs.real]
    if zs_type == "|":
        mmds = []
        for zt in self.zs_type_mmds.keys():
            mmds += self.line_mmds(zt)
        return list(set(mmds))
    if zs_type == "&":
        mmds = self.line_mmds()
        for zt in self.zs_type_mmds.keys():
            mmds = set(mmds) & set(self.line_mmds(zt))
        return list(mmds)
    if zs_type not in self.zs_type_mmds.keys():
        return []
    return [m.name for m in self.zs_type_mmds[zs_type] if m.zs.real]


def _line_line_bcs(self, zs_type=None):
    """返回当前线所有的背驰类型"""
    if zs_type is None:
        return [_bc.type for _bc in self.bcs if _bc.bc and (_bc.zs is None or _bc.zs.real)]
    if zs_type == "|":
        bcs = []
        for zt in self.zs_type_bcs.keys():
            bcs += self.line_bcs(zt)
        return list(set(bcs))
    if zs_type == "&":
        bcs = self.line_bcs()
        for zt in self.zs_type_bcs.keys():
            bcs = set(bcs) & set(self.line_bcs(zt))
        return list(bcs)
    if zs_type not in self.zs_type_bcs.keys():
        return []
    return [_bc.type for _bc in self.zs_type_bcs[zs_type]
            if _bc.bc and (_bc.zs is None or _bc.zs.real)]


def _line_mmd_exists(self, check_mmds, zs_type=None):
    """检查当前线是否包含指定的买卖点中的一个"""
    mmds = self.line_mmds(zs_type)
    return len(set(check_mmds) & set(mmds)) > 0


def _line_bc_exists(self, bc_types, zs_type=None):
    """检查是否有背驰的情况"""
    bcs = self.line_bcs(zs_type)
    return len(set(bc_types) & set(bcs)) > 0


# ==================== 背驰与买卖点（ 反编译） ====================

def query_macd_ld(cd, start_fx, end_fx):
    """计算分型区间 macd 力度（query_macd_ld）"""
    if start_fx.k_index > end_fx.k_index:
        raise Exception(
            "%s - %s - %s 计算力度，开始分型不可以大于结束分型"
            % (cd.code, cd.frequency, cd.src_klines[-1]["t"])
        )
    dea = np.array(cd.get_idx()["macd"]["dea"][start_fx.k_index: end_fx.k_index + 1])
    dif = np.array(cd.get_idx()["macd"]["dif"][start_fx.k_index: end_fx.k_index + 1])
    hist = np.array(cd.get_idx()["macd"]["hist"][start_fx.k_index: end_fx.k_index + 1])
    if len(hist) == 0:
        hist = np.array([0])
    if len(dea) == 0:
        dea = np.array([0])
    if len(dif) == 0:
        dif = np.array([0])
    hist_abs = abs(hist)
    hist_up = np.array([_i for _i in hist if _i > 0])
    hist_down = np.array([_i for _i in hist if _i < 0])
    hist_max = np.max(hist)
    hist_min = np.min(hist)
    hist_sum = hist_abs.sum()
    hist_up_sum = hist_up.sum()
    hist_down_sum = abs(hist_down.sum())
    end_dea = dea[-1]
    end_dif = dif[-1]
    end_hist = hist[-1]
    return {
        "dea": {"end": end_dea, "max": np.max(dea), "min": np.min(dea)},
        "dif": {"end": end_dif, "max": np.max(dif), "min": np.min(dif)},
        "hist": {
            "sum": hist_sum,
            "up_sum": hist_up_sum,
            "down_sum": hist_down_sum,
            "max": hist_max,
            "min": hist_min,
            "end": end_hist,
        },
    }


def compare_ld_beichi(one_ld, two_ld, line_direction):
    """比较两个力度，后者小于前者，返回 True"""
    hist_key = "sum"
    if line_direction == "up":
        hist_key = "up_sum"
    elif line_direction == "down":
        hist_key = "down_sum"
    if "macd" not in two_ld.keys() or "macd" not in one_ld.keys():
        return False
    if two_ld["macd"]["hist"][hist_key] < one_ld["macd"]["hist"][hist_key]:
        return True
    else:
        return False


def user_custom_mmd(cd, line, lines, zs_type, zss):
    """用户可自定义买卖点（cl_interface.py 明文）"""
    if len(lines) < 4 or len(zss) == 0:
        return False
    # 类二类买卖点
    pre_same_line = lines[line.index - 2]
    for mmd in pre_same_line.get_mmds(zs_type):
        if line.type == "down" and mmd.name == "2buy" and line.low > pre_same_line.low:
            new_zss = cd.create_dn_zs("", lines[-4:])
            if len(new_zss) != 1:
                continue
            line.add_mmd("l2buy", new_zss[0], zs_type,
                         msg="二买后，重叠中枢后，不创二买低点，形成类二买")
        if line.type == "up" and mmd.name == "2sell" and line.high < pre_same_line.high:
            new_zss = cd.create_dn_zs("", lines[-4:])
            if len(new_zss) != 1:
                continue
            line.add_mmd("l2sell", new_zss[0], zs_type,
                         msg="二卖后，重叠中枢后，不创二卖高点，形成类二卖")
    # 类三类买卖点
    for mmd in pre_same_line.get_mmds(zs_type):
        if (line.type == "down" and mmd.name == "3buy"
                and line.low > mmd.zs.zg
                and compare_ld_beichi(pre_same_line.get_ld(cd), line.get_ld(cd), "down")):
            new_zss = cd.create_dn_zs("", lines[-4:])
            if len(new_zss) != 1:
                continue
            line.add_mmd("l3buy", new_zss[0], zs_type,
                         msg="三买后，重叠中枢后，不创三买低点，形成类三买")
        if (line.type == "up" and mmd.name == "3sell"
                and line.high < mmd.zs.zd
                and compare_ld_beichi(pre_same_line.get_ld(cd), line.get_ld(cd), "up")):
            new_zss = cd.create_dn_zs("", lines[-4:])
            if len(new_zss) != 1:
                continue
            line.add_mmd("l3sell", new_zss[0], zs_type,
                         msg="三卖后，重叠中枢后，不创三卖高点，形成类三卖")
    return True


def beichi_line(self, pre_line, now_line):
    """线的背驰（ beichi_line）"""
    if pre_line.type != now_line.type:
        return False
    if pre_line.type == "up":
        if now_line.high < pre_line.high or now_line.low < pre_line.low:
            return False
    elif pre_line.type == "down":
        if now_line.low > pre_line.low or now_line.high > pre_line.high:
            return False
    return compare_ld_beichi(pre_line.get_ld(self), now_line.get_ld(self), now_line.type)


def beichi_pz(self, zs, now_line):
    """盘整背驰（ beichi_pz）"""
    if zs.lines[-1].index != now_line.index:
        return (False, None)
    if zs.type not in ("up", "down"):
        return (False, None)
    if now_line.type == "up":
        if now_line.high < max([l.high for l in zs.lines]):
            return (False, None)
    elif now_line.type == "down":
        if now_line.low > min([l.low for l in zs.lines]):
            return (False, None)
    v137 = compare_ld_beichi(zs.lines[0].get_ld(self), now_line.get_ld(self), now_line.type)
    return (v137, zs.lines[0])


def beichi_qs(self, lines, zss, now_line):
    """趋势背驰（ beichi_qs）"""
    if len(zss) < 2:
        return (False, [])
    v138 = zss[-2]
    v139 = zss[-1]
    v140 = self.zss_is_qs(v138, v139)
    if v140 is None:
        return (False, [])
    if now_line.type != v139.lines[0].type:
        return (False, [])
    if now_line.type == "up" and now_line.high <= v139.gg:
        return (False, [])
    if now_line.type == "down" and now_line.low >= v139.dd:
        return (False, [])
    v141 = lines[v138.lines[-1].index: v139.lines[1].index]
    v142 = lines[v139.lines[-1].index: now_line.index + 1]
    v143, v144 = self._lines_level_ld(v141)
    v145, v146 = self._lines_level_ld(v142)
    if v143 is None or v144 is None or v145 is None or v146 is None:
        return (False, [])
    if self.config["judge_zs_qs_level"]:
        if v143 != v145:
            return (False, [])
    v137 = compare_ld_beichi({"macd": v144}, {"macd": v146}, now_line.type)
    return (v137, v141)


def _lines_level_ld(self, lines):
    """线的级别与力度（ _lines_level_ld）"""
    if len(lines) == 0:
        return (None, None)
    if lines[0].type != lines[-1].type:
        return (None, None)
    v226 = max([l.high for l in lines])
    v227 = min([l.low for l in lines])
    if lines[0].type == "up" and lines[0].low != v227:
        return (None, None)
    if lines[0].type == "down" and lines[0].high != v226:
        return (None, None)
    if lines[-1].type == "up" and lines[-1].high != v226:
        return (None, None)
    if lines[-1].type == "down" and lines[-1].low != v227:
        return (None, None)
    if len(lines) == 1:
        return (1, query_macd_ld(self, lines[0].start, lines[-1].end))
    v228 = 0
    v229 = [lines[1].high, lines[1].low]
    for v107 in lines:
        if self._cross_qujian(v229, [v107.high, v107.low]) is not None:
            v228 += 1
    return (int(v228 / 3), query_macd_ld(self, lines[0].start, lines[-1].end))


def process_mmd(self, line_types=None):
    """计算背驰与买卖点（ process_mmd 完整反编译）"""
    if line_types is None:
        return False
    for v122 in line_types:
        if v122 == "bi":
            v123 = self.bis
            v124 = self.type_bi_zss
        elif v122 == "xd":
            v123 = self.xds
            v124 = self.type_xd_zss
        elif v122 == "zsd":
            v123 = self.zsds
            v124 = self.type_zsd_zss
        elif v122 == "qsd":
            v123 = self.qsds
            v124 = {"zs_type_bz": self.qsd_zss}
        else:
            raise Exception("不支持的买卖点类型 %s" % v122)

        if len(v123) == 0:
            return False

        v121 = v123[-1]  # 当前线（只计算最后一条线）
        v121.bcs = []
        v121.mmds = []
        v121.zs_type_mmds = {}
        v121.zs_type_bcs = {}
        for v103 in v124.keys():
            v121.zs_type_mmds[v103] = []
            v121.zs_type_bcs[v103] = []

        for v103, v7 in v124.items():
            v7 = [zs for zs in v7 if zs.real]
            if len(v7) == 0 or len(v123) < 3:
                return True
            v10 = v7[-1]
            v125 = None
            if len(v7) >= 2:
                v125 = self.zss_is_qs(v7[-2], v10)

            # 线的背驰
            v121.add_bc(v122, None, v123[-3], [], self.beichi_line(v123[-3], v121),
                        zs_type=v103)
            # 盘整背驰
            v126, v127 = self.beichi_pz(v10, v121)
            if v126:
                v121.add_bc("pz", v10, v127, [], v126, zs_type=v103)
            # 趋势背驰
            v128, v129 = self.beichi_qs(v123, v7, v121)
            if v128:
                v121.add_bc("qs", v10, None, v129, v128, zs_type=v103)

            # 趋势一卖/一买（两中枢趋势）
            if self.config["cl_mmd_cal_qs_1mmd"]:
                for v130 in v121.zs_type_bcs[v103]:
                    if v130.type == "qs" and v130.bc:
                        if v121.type == "up":
                            v121.add_mmd("1sell", v130.zs, v103, msg="两中枢趋势一卖")
                        if v121.type == "down":
                            v121.add_mmd("1buy", v130.zs, v103, msg="两中枢趋势一买")

            # 非趋势，三卖/三买后背驰一买/一卖
            if self.config["cl_mmd_cal_not_qs_3mmd_1mmd"] and v125 is None:
                v131 = v123[v121.index - 1]
                for v132 in v131.get_mmds(v103):
                    if v121.type == "down" and v132.name == "3sell" \
                            and v132.zs.lines[0].type == "down" \
                            and v121.bc_exists([v122], v103):
                        v121.add_mmd("1buy", v10, v103, msg="非趋势，三卖后背驰一买")
                    if v121.type == "up" and v132.name == "3buy" \
                            and v132.zs.lines[0].type == "up" \
                            and v121.bc_exists([v122], v103):
                        v121.add_mmd("1sell", v10, v103, msg="非趋势，三买后背驰一卖")

            # 趋势，三卖/三买后背驰一买/一卖
            if self.config["cl_mmd_cal_qs_3mmd_1mmd"] and v125 is not None:
                v131 = v123[v121.index - 1]
                for v132 in v131.get_mmds(v103):
                    if v125 == "down" and v121.type == "down" and v132.name == "3sell" \
                            and v132.zs.lines[0].type == "down" \
                            and v121.bc_exists([v122], v103):
                        v121.add_mmd("1buy", v10, v103, msg="趋势，三卖后背驰一买")
                    if v125 == "up" and v121.type == "up" and v132.name == "3buy" \
                            and v132.zs.lines[0].type == "up" \
                            and v121.bc_exists([v122], v103):
                        v121.add_mmd("1sell", v10, v103, msg="趋势，三买后背驰一卖")

            # 趋势二卖/二买（v125 有趋势方向时）
            if v125 is not None:
                v133 = v123[-3]
                v131 = v123[-2]
                if v125 == "up" and v10.lines[0].type == "up" and v121.type == "up":
                    if self.config["cl_mmd_cal_qs_not_lh_2mmd"]:
                        if len(v10.lines) >= 7 and v133.high == v10.gg \
                                and v133.high > v121.high:
                            v121.add_mmd("2sell", v10, v103, msg="趋势后，不创新高后的二卖")
                    if self.config["cl_mmd_cal_qs_bc_2mmd"]:
                        if len(v10.lines) >= 7 and v133.high == v10.gg \
                                and v121.bc_exists([v122], v103):
                            v121.add_mmd("2sell", v10, v103, msg="趋势后，新高后的背驰二卖")
                    if self.config["cl_mmd_cal_3mmd_not_lh_bc_2mmd"]:
                        if v131.mmd_exists(["3buy"], v103):
                            if v133.high > v121.high or v121.bc_exists([v122], v103):
                                v121.add_mmd("2sell", v10, v103,
                                             msg="趋势后，三买后不创新高或背驰的二卖")
                if v125 == "down" and v10.lines[0].type == "down" and v121.type == "down":
                    if self.config["cl_mmd_cal_qs_not_lh_2mmd"]:
                        if len(v10.lines) >= 7 and v133.low == v10.dd \
                                and v133.low < v121.low:
                            v121.add_mmd("2buy", v10, v103, msg="趋势后，不创新低后的二买")
                    if self.config["cl_mmd_cal_qs_bc_2mmd"]:
                        if len(v10.lines) >= 7 and v133.low == v10.dd \
                                and v121.bc_exists([v122], v103):
                            v121.add_mmd("2buy", v10, v103, msg="趋势后，新低后的背驰二买")
                    if self.config["cl_mmd_cal_3mmd_not_lh_bc_2mmd"]:
                        if v131.mmd_exists(["3sell"], v103):
                            if v133.low < v121.low or v121.bc_exists([v122], v103):
                                v121.add_mmd("2buy", v10, v103,
                                             msg="趋势后，三卖后不创新高或背驰的二买")

            # 一买/一卖后，不创新低/高的二买/二卖
            if self.config["cl_mmd_cal_1mmd_not_lh_2mmd"]:
                v133 = v123[-3]
                for v132 in v133.get_mmds(v103):
                    if v132.name == "1buy" and v121.type == "down" and v121.low > v133.low:
                        v121.add_mmd("2buy", v132.zs, v103, msg="一类买点后，不创新低二买")
                    if v132.name == "1sell" and v121.type == "up" and v121.high < v133.high:
                        v121.add_mmd("2sell", v132.zs, v103, msg="一类卖点后，不创新高二卖")

            # 三买/三卖创新高不背驰，小转大二卖/二买信号
            if self.config["cl_mmd_cal_3mmd_xgxd_not_bc_2mmd"]:
                if len(v123) > 4:
                    v134 = v123[-4]
                    v133 = v123[-3]
                    for v132 in v134.get_mmds(v103):
                        if v132.name == "3buy" and v121.type == "up" \
                                and v121.high < v133.high \
                                and compare_ld_beichi(v133.get_ld(self), v121.get_ld(self), v121.type) \
                                and not v133.mmd_exists(["1sell"], v103):
                            v121.add_mmd("2sell", v132.zs, v103,
                                         msg="三类买点创新高不背驰产生一卖，小转大二卖信号")
                        if v132.name == "3sell" and v121.type == "down" \
                                and v121.low > v133.low \
                                and compare_ld_beichi(v133.get_ld(self), v121.get_ld(self), v121.type) \
                                and not v133.mmd_exists(["1buy"], v103):
                            v121.add_mmd("2buy", v132.zs, v103,
                                         msg="三类卖点创新低不背驰产生一买，小转大二买信号")

            # 中枢外三买/三卖（小于9段）
            if self.config["cl_mmd_cal_not_in_zs_3mmd"]:
                v135 = [zs for zs in v7 if zs.real and zs.done
                        and len(zs.lines) < 9
                        and zs.lines[-1].end.index == v121.start.index]
                for v136 in v135:
                    if v136.lines[-1].index + 1 == v121.index:
                        if v121.type == "up" and v121.high < v136.zd:
                            v121.add_mmd("3sell", v136, v103,
                                         msg="中枢下方(小于9段)，向上段不高于中枢低点后的三卖")
                        if v121.type == "down" and v121.low > v136.zg:
                            v121.add_mmd("3buy", v136, v103,
                                         msg="中枢上方(小于9段)，向下段不低于中枢高点后的三买")

            # 中枢外三买/三卖（大于等于9段）
            if self.config["cl_mmd_cal_not_in_zs_gt_9_3mmd"]:
                v135 = [zs for zs in v7 if zs.real and zs.done
                        and len(zs.lines) >= 9
                        and zs.lines[-1].end.index == v121.start.index]
                for v136 in v135:
                    if v136.lines[-1].index + 1 == v121.index:
                        if v121.type == "up" and v121.high < v136.zd:
                            v121.add_mmd("3sell", v136, v103,
                                         msg="中枢下方(大于9段)，向上段不高于中枢低点后的三卖")
                        if v121.type == "down" and v121.low > v136.zg:
                            v121.add_mmd("3buy", v136, v103,
                                         msg="中枢上方(大于9段)，向下段不低于中枢高点后的三买")

            # 用户自定义买卖点
            user_custom_mmd(self, v121, v123, v103, v7)

    return True


def process_idx(self):
    """计算 macd 指标（ process_idx，TA-Lib MACD + 增量拼接）"""
    v24 = ["idx_macd_fast", "idx_macd_slow", "idx_macd_signal"]
    v25 = 0
    for v2 in v24:
        v25 = max(v25, self.config[v2])
    v4 = self.src_klines
    v25 = 500 + v25
    v25 = v25 if v25 <= len(v4) else len(v4)
    v27 = [k["c"] for k in v4[-v25:]]
    v28, v29, v30 = ta.MACD(np.array(v27), self.config["idx_macd_fast"],
                            self.config["idx_macd_slow"], self.config["idx_macd_signal"])
    v30 *= 2
    if len(self.idx["macd"]["hist"]) == 0 or len(v4) == v25:
        self.idx["macd"]["dea"] = v29
        self.idx["macd"]["dif"] = v28
        self.idx["macd"]["hist"] = v30
    else:
        v31 = len(v4) - len(self.idx["macd"]["hist"])
        self.idx["macd"]["dea"] = np.concatenate(
            (np.array(self.idx["macd"]["dea"][0:-1]), np.array(v29[-(v31 + 1):])), 0)
        self.idx["macd"]["dif"] = np.concatenate(
            (np.array(self.idx["macd"]["dif"][0:-1]), np.array(v28[-(v31 + 1):])), 0)
        self.idx["macd"]["hist"] = np.concatenate(
            (np.array(self.idx["macd"]["hist"][0:-1]), np.array(v30[-(v31 + 1):])), 0)
    return True


def get_idx(self):
    """返回指标数据"""
    return self.idx


CL._add_time = _add_time
CL.process_zs = process_zs
CL.process_mmd = process_mmd
CL._process_bz_zs = _process_bz_zs
CL.create_dn_zs = create_dn_zs
CL.zss_is_qs = zss_is_qs
CL.create_zs_list = create_zs_list
CL.create_zs = create_zs
CL._line_get_zs_hl = _line_get_zs_hl
CL.get_ld = _line_get_ld
CL.beichi_line = beichi_line
CL.beichi_pz = beichi_pz
CL.beichi_qs = beichi_qs
CL._lines_level_ld = _lines_level_ld
CL.process_idx = process_idx
CL.get_idx = get_idx
BI.get_ld = _line_get_ld
BI.get_mmds = _line_get_mmds
BI.add_mmd = _line_add_mmd
BI.add_bc = _line_add_bc
BI.line_mmds = _line_line_mmds
BI.line_bcs = _line_line_bcs
BI.mmd_exists = _line_mmd_exists
BI.bc_exists = _line_bc_exists
XD.get_ld = _line_get_ld
XD.get_mmds = _line_get_mmds
XD.add_mmd = _line_add_mmd
XD.add_bc = _line_add_bc
XD.line_mmds = _line_line_mmds
XD.line_bcs = _line_line_bcs
XD.mmd_exists = _line_mmd_exists
XD.bc_exists = _line_bc_exists


# ==================== chanlun.cl 兼容层（替换cl.py 用） ====================
class KDict(dict):
    """原始K线 dict，同时支持CK 的属性访问（.date/.h/.l/.o/.c/.v/.a/.index）"""
    __slots__ = ()

    def __getattr__(self, name):
        if name == "date":
            return self["t"]
        if name == "a":
            return self["v"]
        if name in self:
            return self[name]
        raise AttributeError(name)


def _cl_get_src_klines(self):
    return self.src_klines


def _cl_get_klines(self):
    # get_klines 返回原始K线（src），且 cl_utils 用 FX.k.k_index 索引本列表
    return self.src_klines


def _cl_get_code(self):
    return self.code


def _cl_get_frequency(self):
    return self.frequency


def _cl_get_fxs(self):
    return self.fxs


def _cl_get_bis(self):
    return self.bis


def _cl_get_xds(self):
    return self.xds


def _cl_get_zsds(self):
    return self.zsds


def _cl_get_qsds(self):
    return self.qsds


def _cl_get_bi_zss(self, zs_type=None):
    if zs_type is None:
        return self.bi_zss
    return self.type_bi_zss.get(zs_type, [])


def _cl_get_xd_zss(self, zs_type=None):
    if zs_type is None:
        return self.xd_zss
    return self.type_xd_zss.get(zs_type, [])


def _cl_get_zsd_zss(self):
    return self.zsd_zss


def _cl_get_qsd_zss(self):
    return self.qsd_zss


def _cl_get_config(self):
    return self.config


CL.get_src_klines = _cl_get_src_klines
CL.get_klines = _cl_get_klines
CL.get_code = _cl_get_code
CL.get_frequency = _cl_get_frequency
CL.get_fxs = _cl_get_fxs
CL.get_bis = _cl_get_bis
CL.get_xds = _cl_get_xds
CL.get_zsds = _cl_get_zsds
CL.get_qsds = _cl_get_qsds
CL.get_bi_zss = _cl_get_bi_zss
CL.get_xd_zss = _cl_get_xd_zss
CL.get_zsd_zss = _cl_get_zsd_zss
CL.get_qsd_zss = _cl_get_qsd_zss
CL.get_config = _cl_get_config
