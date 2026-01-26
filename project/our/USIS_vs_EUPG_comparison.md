# USIS-SAM vs EUPG — Short Comparison

## Summary
Short comparison of the two designs in this workspace: `USIS-SAM` (in `USIS10K/project/our`) and `EUPG` (in `uwsam_teacher`).

## Comparison table

- **Prompt generator location**: USIS — inside mask head (`point_emb`); EUPG — separate `EUPGHead` module.
- **Dense prompt**: USIS — uses SAM's `no_mask_embed` constant; EUPG — passes `image_pe` and uses EUPG prompts, SAM prompt encoder not required.
- **Modularity**: USIS — compact (decoder + prompt creation co-located). EUPG — modular, end-to-end prompt generator.
- **Training flow**: USIS — standard Mask R-CNN flow with mask head generating prompts per-ROI. EUPG — RPN -> `EUPGHead` (assignment + prompts) -> SAM decoder wrapper.
- **Multi-scale**: both implement 2x and 4x upsampling for RPN features.

## Quick code highlights (open links)

- USIS-SAM
  - Mask head config: [project/our/configs/anchor_net.py](project/our/configs/anchor_net.py#L147)
  - Build `mask_decoder` in mask head: [project/our/our_model/anchor.py](project/our/our_model/anchor.py#L593)
  - Extract `no_mask_embed` (dense prompt): [project/our/our_model/anchor.py](project/our/our_model/anchor.py#L602)
  - Create `dense_embeddings` (reshape/expand): [project/our/our_model/anchor.py](project/our/our_model/anchor.py#L649)
  - Call SAM mask decoder (passes sparse + dense prompts): [project/our/our_model/anchor.py](project/our/our_model/anchor.py#L655)
  - FPN upsampling (2x, 4x): [project/our/our_model/anchor.py](project/our/our_model/anchor.py#L188-L199)
  - SAM wrappers: `USISSamPromptEncoder` / `USISSamMaskDecoder` — [project/our/our_model/common.py](project/our/our_model/common.py#L153), [project/our/our_model/common.py](project/our/our_model/common.py#L221)

- EUPG (uwsam_teacher)
  - `EUPGHead` prompt projection: [uwsam_teacher/eupg_head.py](uwsam_teacher/eupg_head.py#L156)
  - `EUPGHead.forward` returns prompts and `image_pe`: [uwsam_teacher/eupg_head.py](uwsam_teacher/eupg_head.py#L177)
  - SAM wrapper `SAMHeadV2` (expects external prompts): [uwsam_teacher/sam_head_v2.py](uwsam_teacher/sam_head_v2.py#L15) and forward: [uwsam_teacher/sam_head_v2.py](uwsam_teacher/sam_head_v2.py#L61)

## Recommendation (one line)
- Use USIS-SAM for simpler, compact integration; use EUPG structure when you want a modular, end-to-end trainable prompt generator to experiment with different prompt strategies.
