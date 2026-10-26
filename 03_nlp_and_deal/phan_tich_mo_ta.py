TEN_CO = {
    "vb_chinh_chu": "Chính chủ / một chủ",
    "vb_bao_duong_hang": "Bảo dưỡng hãng",
    "vb_khong_dam_dung": "Cam kết không đâm đụng",
    "vb_khong_ngap_nuoc": "Cam kết không ngập nước",
    "vb_zin": "Zin / nguyên bản",
    "vb_khong_tua_odo": "Cam kết không tua ODO",
    "vb_bao_test": "Cho kiểm tra / bao test",
    "vb_tra_gop": "Hỗ trợ trả góp",
    "vb_bao_hanh": "Có nhắc bảo hành",
}


def tom_tat_mo_ta(thong_tin):
    cam_ket = [ten for cot, ten in TEN_CO.items() if thong_tin.get(cot, 0) == 1]
    return {
        "co_mo_ta": thong_tin.get("co_mo_ta", 0) == 1,
        "so_cam_ket": len(cam_ket),
        "cam_ket": cam_ket,
        "noi_dung": ", ".join(cam_ket) if cam_ket else "Không tìm thấy cam kết",
    }
