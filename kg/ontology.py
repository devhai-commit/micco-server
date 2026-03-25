from enum import Enum


class NodeLabel(str, Enum):
    # Domain entities (17)
    VAT_TU          = "VatTu"
    NHA_CUNG_CAP    = "NhaCungCap"
    HOP_DONG        = "HopDong"
    KHO             = "Kho"
    QUY_DINH        = "QuyDinh"
    SU_CO           = "SuCo"
    DON_HANG        = "DonHang"
    KE_HOACH        = "KeHoachMuaSam"
    CHAO_GIA        = "ChaoGia"
    PHIEU_NHAP      = "PhieuNhapKho"
    PHIEU_XUAT      = "PhieuXuatKho"
    CONG_TRUONG     = "CongTruong"
    NGUOI_KIEM_TRA  = "NguoiKiemTra"
    CHUNG_CHI       = "ChungChi"
    DANH_MUC        = "DanhMucVatTu"
    NHA_SAN_XUAT    = "NhaSanXuat"
    RUI_RO          = "RuiRo"
    # Knowledge base (1)
    TRI_THUC        = "TriThuc"
    # Structural node (1)
    DOCUMENT        = "Document"


class RelType(str, Enum):
    # Procurement chain (4)
    BAO_GOM             = "BAO_GOM"
    CUNG_CAP            = "CUNG_CAP"
    TU_NHA_CUNG_CAP     = "TU_NHA_CUNG_CAP"
    DUOC_LAP_TU         = "DUOC_LAP_TU"
    # Storage & logistics (3)
    NHAP_VAT_TU         = "NHAP_VAT_TU"
    XUAT_DEN            = "XUAT_DEN"
    GIAO_HANG           = "GIAO_HANG"
    # Quality & compliance (2)
    TUAN_THU_THEO       = "TUAN_THU_THEO"
    CO_CHUNG_CHI        = "CO_CHUNG_CHI"
    # Risk & incident (3)
    LIEN_QUAN_DEN       = "LIEN_QUAN_DEN"
    LIEN_QUAN           = "LIEN_QUAN"
    LIEN_QUAN_VAT_TU    = "LIEN_QUAN_VAT_TU"
    # Traceability (3)
    NGUON_GOC           = "NGUON_GOC"
    SAN_XUAT_BOI        = "SAN_XUAT_BOI"
    THUOC_DANH_MUC      = "THUOC_DANH_MUC"
    # Procurement lifecycle (5)
    CO_DON_HANG         = "CO_DON_HANG"
    YEU_CAU_CHAO_GIA    = "YEU_CAU_CHAO_GIA"
    DUOC_KY_KET         = "DUOC_KY_KET"
    LUU_TAI             = "LUU_TAI"
    XUAT_TU_KHO         = "XUAT_TU_KHO"
    # Document authorship & approval (5)
    DO_NGUOI_LAP        = "DO_NGUOI_LAP"
    DUOC_DUYET_BOI      = "DUOC_DUYET_BOI"
    DUOC_KIEM_TRA_BOI   = "DUOC_KIEM_TRA_BOI"
    THUOC_PHONG_BAN     = "THUOC_PHONG_BAN"
    AP_DUNG_CHO         = "AP_DUNG_CHO"
    # Material properties (5)
    CO_DANH_MUC         = "CO_DANH_MUC"
    SU_DUNG_TAI         = "SU_DUNG_TAI"
    THAY_THE_BOI        = "THAY_THE_BOI"
    TUONG_DUONG_VOI     = "TUONG_DUONG_VOI"
    CO_RUI_RO           = "CO_RUI_RO"
    # Risk & compliance (5)
    DUOC_PHAT_HIEN_TAI  = "DUOC_PHAT_HIEN_TAI"
    XAY_RA_TAI          = "XAY_RA_TAI"
    DUOC_XU_LY_BOI      = "DUOC_XU_LY_BOI"
    BAO_CAO_TAI         = "BAO_CAO_TAI"
    LIEU_QUAN_HOP_DONG  = "LIEU_QUAN_HOP_DONG"
    # Certificate lifecycle (2)
    CAP_BOI             = "CAP_BOI"
    XAC_NHAN_CHAT_LUONG = "XAC_NHAN_CHAT_LUONG"
