# Face fixture credits

All three images are official White House portraits, in the public domain in the United States as works of
federal government employees made as part of their official duties (Wikimedia Commons licence tag
`PD-USGov-POTUS`; Commons reports `License: pd`, `Copyrighted: False`). They are used only as test
fixtures. Public domain status covers copyright only; it does not imply any endorsement by the people
shown, and Commons notes that personality rights may apply to some uses.

Each file is a crop of the 1280 px wide rendition that Commons serves. The crop is a 4:3 head-and-shoulders
region, resized to 640 × 480 (OpenCV `INTER_AREA`) and saved as JPEG quality 90 with metadata removed. The
640 × 480 size matches the fake camera in `tests/e2e/helpers/fakeCamera.js`, which then shows the whole image
without cropping it. Crop rectangles are `x, y, width, height` in pixels of the 1280 px rendition.

| File | Person | Photo | Author | Source |
|---|---|---|---|---|
| `a1.jpg` | Joe Biden (person A) | Official presidential portrait, 3 March 2021 | Adam Schultz, Official White House Photo | https://commons.wikimedia.org/wiki/File:Joe_Biden_presidential_portrait.jpg |
| `a2.jpg` | Joe Biden (person A) | Official vice-presidential portrait, 10 January 2013 | David Lienemann, Official White House Photo | https://commons.wikimedia.org/wiki/File:Joe_Biden_official_portrait_2013.jpg |
| `b1.jpg` | Barack Obama (person B) | Official presidential portrait, 6 December 2012 | Pete Souza, Official White House Photo | https://commons.wikimedia.org/wiki/File:President_Barack_Obama.jpg |

## Details

- **a1.jpg**
  - Downloaded from https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Joe_Biden_presidential_portrait.jpg/1280px-Joe_Biden_presidential_portrait.jpg (1280 × 1600)
  - Original source, as given on Commons: https://www.whitehouse.gov/administration/president-biden/ (archived at https://bidenwhitehouse.archives.gov/administration/president-biden/)
  - Crop: `0, 128, 1280, 960`
- **a2.jpg**
  - Downloaded from https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Joe_Biden_official_portrait_2013.jpg/1280px-Joe_Biden_official_portrait_2013.jpg (1280 × 1600)
  - Original source, as given on Commons: https://www.flickr.com/photos/whitehouse/8508244102 (White House, V011013DL-0556)
  - Crop: `304, 26, 908, 681`
- **b1.jpg**
  - Downloaded from https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/1280px-President_Barack_Obama.jpg (1280 × 1599)
  - Original source, as given on Commons: https://www.flickr.com/photos/whitehouse/8390033709/ (White House, P120612PS-0463)
  - Crop: `107, 0, 1062, 796`

`reference.json` comes from `scripts/reference_embeddings.py` (OpenCV YuNet + FaceRecognizerSF, fp32 SFace).
`js-embeddings.json` comes from the browser pipeline (`UPDATE_FIXTURES=1 npx playwright test tests/e2e/faceEmbedding.spec.js`).
