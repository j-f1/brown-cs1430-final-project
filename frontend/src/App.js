// @ts-check
import "./App.css";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";
import React, { useCallback, useRef, useState } from "react";
import { Button } from "react-bootstrap";

const SERVER = "http://localhost.proxyman.io:5000";

function App() {
  const [image, setImage] = useState(null);
  const [faceRects, setFaceRects] = useState(null);
  /** @type {React.MutableRefObject<HTMLFormElement>} */
  const formRef = useRef();

  const onSelect = useCallback(() => {
    fetch(SERVER + "/select-image", {
      method: "POST",
      body: new FormData(formRef.current),
    })
      .then((res) => res.json())
      .then(({ image, faces }) => {
        setImage(image);
        setFaceRects(faces);
      });
  }, []);

  const onSwap = useCallback((e) => {
    e.preventDefault();
    fetch(SERVER + "/swap", {
      method: "POST",
      body: image,
    })
      .then((res) => res.json())
      .then(({ image }) => {
        console.log(image);
      });
  }, []);

  return (
    <Row>
      <Col lg={1} />
      <Col md={6} lg={5} className="p-4">
        <h2>Image Input</h2>
        {image ? (
          <>
            <div style={{ position: "relative" }}>
              <img
                src={"data:;base64," + image}
                alt=""
                style={{ width: "100%" }}
              />
              {faceRects.map((faceRect, i) => (
                <div
                  key={i}
                  style={{
                    position: "absolute",
                    top: faceRect.y * 100 + "%",
                    left: faceRect.x * 100 + "%",
                    width: faceRect.width * 100 + "%",
                    height: faceRect.height * 100 + "%",
                    border: "1px solid red",
                    cursor: "pointer",
                  }}
                  onClick={(e) => {
                    e.preventDefault();
                    console.log("click", i, faceRect);
                    // fetch(SERVER + "/select-face", {
                    //   method: "POST",
                    //   body: JSON.stringify({
                    //     image,
                    //     face: i,
                    //   }),
                    // });
                  }}
                />
              ))}
            </div>
          </>
        ) : (
          <Form ref={formRef}>
            <Form.Group controlId="image-upload">
              <Form.Label>Image</Form.Label>
              <Form.Control
                name="image"
                type="file"
                size="sm"
                onChange={onSelect}
              />
            </Form.Group>
          </Form>
        )}
        <Form onSubmit={onSwap}>
          <Button className="mt-2" size="sm" type="submit">
            Swap Faces
          </Button>
        </Form>
      </Col>
      <Col md={6} lg={5}>
        A
      </Col>
      <Col lg={1} />
    </Row>
  );
}

export default App;
