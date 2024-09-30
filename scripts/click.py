import os.path
from PIL import Image, ImageDraw
import cv2


class Click:
    def __init__(self, circle_radius_945=15):
        self.orig_image = None
        self.image_cv2 = None
        self.x_g = self.y_g = None
        self.circle_radius_945 = circle_radius_945
        self.original_cv2 = None

    def create_circle_around_point(
        self, original_image, row, column, circle_radius_945
    ):
        new_image = Image.new("RGB", original_image.size, color="black")
        circle_radius = int(round(original_image.size[0] / 945 * circle_radius_945))
        draw = ImageDraw.Draw(new_image)
        center = (column, row)
        draw.ellipse(
            [
                center[0] - circle_radius,
                center[1] - circle_radius,
                center[0] + circle_radius,
                center[1] + circle_radius,
            ],
            fill="white",
            outline="white",
        )
        return new_image

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_g, self.y_g = x, y
            self.image_cv2 = self.original_cv2.copy()
            cv2.circle(
                self.image_cv2, (x, y), radius=15, color=(255, 255, 0), thickness=2
            )
            cv2.circle(
                self.image_cv2, (x, y), radius=3, color=(255, 255, 0), thickness=-1
            )
            cv2.imshow("image", self.image_cv2)

    def __call__(self, img_path, click_path):
        self.image_cv2 = cv2.resize(cv2.imread(img_path, 1), (512, 512))
        self.original_cv2 = self.image_cv2.copy()
        self.orig_image = Image.open(img_path).resize((512, 512))
        print(
            "Click on desired point (last click will be taken)."
            " Press enter to save and continue."
        )
        cv2.imshow("image", self.image_cv2)

        cv2.setMouseCallback("image", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        row, column = self.y_g, self.x_g
        print("continuing...")

        if row is not None:
            click_path_no_ext = os.path.splitext(click_path)[0]
            for ext in ("JPG", "JPEG", "jpeg", "png", "PNG"):
                candidate = f"{click_path_no_ext}.{ext}"
                if os.path.exists(candidate):
                    os.remove(candidate)

            click = self.create_circle_around_point(
                self.orig_image, row, column, self.circle_radius_945
            )
            self.y_g, self.x_g = None, None
            click.save(click_path, quality=95)

        return click_path
